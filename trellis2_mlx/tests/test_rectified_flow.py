"""Rectified-flow sampler — parity vs a PT reference of the upstream recipe.

The upstream sampler hierarchy is
``FlowEulerSampler`` + ``ClassifierFreeGuidanceSamplerMixin`` +
``GuidanceIntervalSamplerMixin``. None of it touches CUDA — it's all
pure Python + tensor math — but it does call into the user-supplied
``model``. We diff against a self-contained PT port of the same recipe
using a tiny analytic dummy "model" so the test runs in milliseconds.

Coverage:

* ``test_t_schedule_matches_upstream`` — the rescaled timestep linspace.
* ``test_sample_no_cfg`` — guidance_strength=1 path (skip CFG).
* ``test_sample_full_cfg`` — CFG + rescale + interval (all three on).
* ``test_sample_guidance_strength_zero`` — uncond-only path.
* ``test_pred_xstart_round_trip`` — ``v ↔ x0`` round-trip identity.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch

from trellis2_mlx.samplers.rectified_flow import RectifiedFlowSampler, SamplerParams

pytestmark = pytest.mark.reference


# ── PT reference (mirrors the upstream three-class stack) ────────────────


class _PTReferenceSampler:
    """Self-contained PT port of FlowEuler + CFG + Interval mixins."""

    def __init__(self, sigma_min: float = 1e-5) -> None:
        self.sigma_min = sigma_min

    def _pred_to_xstart(self, x_t, t, v):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _cfg_combine(self, x_t, t, v_pos, v_neg, guidance_strength, guidance_rescale):
        v = guidance_strength * v_pos + (1 - guidance_strength) * v_neg
        if guidance_rescale > 0:
            x0_pos = self._pred_to_xstart(x_t, t, v_pos)
            x0_cfg = self._pred_to_xstart(x_t, t, v)
            # Match upstream classifier_free_guidance_mixin.py:23 — std over
            # all dims except batch (axis 0), keepdim so it broadcasts.
            if x0_pos.ndim == 1:
                std_pos = x0_pos.std(keepdim=True)
                std_cfg = x0_cfg.std(keepdim=True)
            else:
                std_pos = x0_pos.std(dim=list(range(1, x0_pos.ndim)), keepdim=True)
                std_cfg = x0_cfg.std(dim=list(range(1, x0_cfg.ndim)), keepdim=True)
            x0_rescale = x0_cfg * (std_pos / (std_cfg + 1e-12))
            x0_final = guidance_rescale * x0_rescale + (1 - guidance_rescale) * x0_cfg
            v = self._xstart_to_pred(x_t, t, x0_final)
        return v

    def _predict_v(
        self,
        model_fn,
        x_t,
        t,
        cond,
        neg_cond,
        guidance_strength,
        guidance_rescale,
        guidance_interval,
        **kw,
    ):
        t_scaled = torch.tensor([1000.0 * t], dtype=torch.float32)
        in_interval = guidance_interval[0] <= t <= guidance_interval[1]
        if not in_interval or guidance_strength == 1.0 or neg_cond is None:
            return model_fn(x_t, t_scaled, cond, **kw)
        if guidance_strength == 0.0:
            return model_fn(x_t, t_scaled, neg_cond, **kw)
        v_pos = model_fn(x_t, t_scaled, cond, **kw)
        v_neg = model_fn(x_t, t_scaled, neg_cond, **kw)
        return self._cfg_combine(x_t, t, v_pos, v_neg, guidance_strength, guidance_rescale)

    def sample(
        self,
        model_fn,
        noise,
        cond,
        neg_cond,
        *,
        params: SamplerParams,
        **kw,
    ):
        t_seq = np.linspace(1.0, 0.0, params.steps + 1)
        t_seq = params.rescale_t * t_seq / (1.0 + (params.rescale_t - 1.0) * t_seq)
        sample = noise
        for t, t_prev in zip(t_seq[:-1], t_seq[1:], strict=False):
            v = self._predict_v(
                model_fn,
                sample,
                float(t),
                cond,
                neg_cond,
                params.guidance_strength,
                params.guidance_rescale,
                params.guidance_interval,
                **kw,
            )
            sample = sample - (t - t_prev) * v
        return sample


# ── Dummy models ─────────────────────────────────────────────────────────


def _mlx_dummy_model(x, t_scaled, cond, *, scale=0.5):
    """v = scale · (cond - x) · (1 + 0.1 · t) — depends on both conditioning
    and timestep so CFG actually pulls in a different direction for
    cond / neg_cond."""
    t_factor = 1.0 + 0.1 * (t_scaled / 1000.0)
    return scale * (cond - x) * t_factor


def _pt_dummy_model(x, t_scaled, cond, *, scale=0.5):
    t_factor = 1.0 + 0.1 * (t_scaled / 1000.0)
    return scale * (cond - x) * t_factor


# ── Tests ────────────────────────────────────────────────────────────────


def test_t_schedule_matches_upstream() -> None:
    """linspace(1, 0, S+1) → rescaled with rescale_t * t / (1 + (r-1) * t)."""
    for r in (1.0, 3.0, 5.0):
        ours = RectifiedFlowSampler.t_schedule(12, r)
        ref_seq = np.linspace(1.0, 0.0, 13)
        ref = (r * ref_seq / (1.0 + (r - 1.0) * ref_seq)).tolist()
        np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_sample_no_cfg() -> None:
    """guidance_strength=1 → no CFG; sampler reduces to pure Euler integration."""
    rng = np.random.default_rng(0)
    shape = (4, 8)
    noise_np = rng.standard_normal(shape).astype(np.float32)
    cond_np = rng.standard_normal(shape).astype(np.float32)

    params = SamplerParams(
        steps=8,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        guidance_interval=(0.0, 1.0),
        rescale_t=1.0,
    )

    mlx_out = np.asarray(
        RectifiedFlowSampler().sample(
            _mlx_dummy_model,
            mx.array(noise_np),
            mx.array(cond_np),
            neg_cond=None,
            params=params,
        )
    )
    ref_out = (
        _PTReferenceSampler()
        .sample(
            _pt_dummy_model,
            torch.from_numpy(noise_np),
            torch.from_numpy(cond_np),
            neg_cond=None,
            params=params,
        )
        .numpy()
    )
    np.testing.assert_allclose(mlx_out, ref_out, atol=1e-6, rtol=1e-6)


def test_sample_full_cfg() -> None:
    """All CFG features on: strength=7.5, rescale=0.5, interval=(0.6, 1.0)."""
    rng = np.random.default_rng(1)
    shape = (4, 8)
    noise_np = rng.standard_normal(shape).astype(np.float32)
    cond_np = rng.standard_normal(shape).astype(np.float32)
    neg_cond_np = np.zeros(shape, dtype=np.float32)

    params = SamplerParams(
        steps=12,
        guidance_strength=7.5,
        guidance_rescale=0.5,
        guidance_interval=(0.6, 1.0),
        rescale_t=3.0,
    )

    mlx_out = np.asarray(
        RectifiedFlowSampler().sample(
            _mlx_dummy_model,
            mx.array(noise_np),
            mx.array(cond_np),
            neg_cond=mx.array(neg_cond_np),
            params=params,
        )
    )
    ref_out = (
        _PTReferenceSampler()
        .sample(
            _pt_dummy_model,
            torch.from_numpy(noise_np),
            torch.from_numpy(cond_np),
            neg_cond=torch.from_numpy(neg_cond_np),
            params=params,
        )
        .numpy()
    )
    diff = np.abs(mlx_out - ref_out)
    msg = f"max={diff.max():.3e}  mean={diff.mean():.3e}"
    assert diff.max() < 1e-5, msg


def test_sample_guidance_strength_zero() -> None:
    """guidance_strength=0 → use uncond only at every step inside the interval."""
    rng = np.random.default_rng(2)
    shape = (4, 8)
    noise_np = rng.standard_normal(shape).astype(np.float32)
    cond_np = rng.standard_normal(shape).astype(np.float32)
    neg_cond_np = rng.standard_normal(shape).astype(np.float32)

    params = SamplerParams(
        steps=4,
        guidance_strength=0.0,
        guidance_rescale=0.0,
        guidance_interval=(0.0, 1.0),
        rescale_t=1.0,
    )

    mlx_out = np.asarray(
        RectifiedFlowSampler().sample(
            _mlx_dummy_model,
            mx.array(noise_np),
            mx.array(cond_np),
            neg_cond=mx.array(neg_cond_np),
            params=params,
        )
    )
    ref_out = (
        _PTReferenceSampler()
        .sample(
            _pt_dummy_model,
            torch.from_numpy(noise_np),
            torch.from_numpy(cond_np),
            neg_cond=torch.from_numpy(neg_cond_np),
            params=params,
        )
        .numpy()
    )
    np.testing.assert_allclose(mlx_out, ref_out, atol=1e-6, rtol=1e-6)


@pytest.mark.slow
def test_sampler_against_real_slat_dit() -> None:
    """End-to-end: real 1.3B SLAT-shape DiT + sampler, 2 steps, no CFG.

    Proves the sampler + DiT compose: noise in, denoised latent out, no
    NaN / inf. Two steps to keep the wall time bounded (each step is a
    full 30-block transformer pass).
    """
    from pathlib import Path

    dit_path = Path("reference/weights/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")
    if not dit_path.exists():
        pytest.skip(f"SLAT DiT weights not found at {dit_path}")

    from trellis2_mlx.models.dit import SLatFlowConfig, SLatFlowModel
    from trellis2_mlx.tests.test_sparse_blocks import _safetensors_load_keys
    from trellis2_mlx.utils.weight_convert import slat_flow_model_from_pt_state_dict

    cfg = SLatFlowConfig()
    model = SLatFlowModel(cfg)

    # Load full DiT state.
    import struct

    with open(dit_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = __import__("json").loads(f.read(n).decode())
    header.pop("__metadata__", None)
    state = _safetensors_load_keys(dit_path, list(header.keys()))
    state = {k: v.astype(np.float32) for k, v in state.items()}
    model.load_weights(slat_flow_model_from_pt_state_dict(state))

    rng = np.random.default_rng(0)
    n_voxels = 8
    n_ctx = 32
    noise = mx.array(rng.standard_normal((n_voxels, cfg.in_channels)).astype(np.float32))
    coords = mx.array(rng.integers(0, cfg.resolution, size=(n_voxels, 3), dtype=np.int32))
    cond = mx.array(rng.standard_normal((1, n_ctx, cfg.cond_channels)).astype(np.float32) * 0.5)

    # Adapt the DiT to the sampler's model_fn signature.
    def model_fn(x, t_scaled, cond_arg, **kw):
        return model(x, kw["coords"], t_scaled, cond_arg)

    out = RectifiedFlowSampler().sample(
        model_fn,
        noise,
        cond=cond,
        neg_cond=None,
        params=SamplerParams(
            steps=2,
            guidance_strength=1.0,
            guidance_rescale=0.0,
            guidance_interval=(0.0, 1.0),
            rescale_t=1.0,
        ),
        coords=coords,
    )
    mx.eval(out)
    out_np = np.asarray(out)

    assert out_np.shape == (n_voxels, cfg.out_channels)
    assert np.isfinite(out_np).all(), "sampler produced non-finite output"
    print(
        f"\n  sampler smoke OK: real 1.3B DiT, 2 steps, no CFG → "
        f"out_range=[{out_np.min():.3f}, {out_np.max():.3f}] std={out_np.std():.3f}"
    )


def test_pred_xstart_round_trip() -> None:
    """v ↔ x0 conversion is exact (modulo fp32 drift)."""
    sampler = RectifiedFlowSampler(sigma_min=1e-5)
    rng = np.random.default_rng(0)
    x_t = mx.array(rng.standard_normal((6, 12)).astype(np.float32))
    v = mx.array(rng.standard_normal((6, 12)).astype(np.float32))
    for t in (0.0, 0.1, 0.5, 0.9, 1.0):
        x_0 = sampler._pred_to_xstart(x_t, t, v)
        # At t=0 the inverse division blows up; skip exact round-trip there.
        if t > 1e-3:
            v_recovered = sampler._xstart_to_pred(x_t, t, x_0)
            np.testing.assert_allclose(np.asarray(v_recovered), np.asarray(v), atol=1e-4, rtol=1e-4)
