"""End-to-end smoke test for the full SLAT DiT generator (1.3B params).

Loads ``slat_flow_img2shape_dit_1_3B_512_bf16.safetensors`` into
:class:`trellis2_mlx.models.dit.SLatFlowModel`, runs one forward pass on
a synthetic latent + DINOv3-shaped context, and verifies:

* The 640 PT parameter keys all map cleanly into the MLX module tree.
* One full forward pass through 30 transformer blocks runs without
  crashing and produces ``[L, 32]`` output (shape latent prediction).
* The output is finite (no NaN / inf — would indicate norm or dtype bugs).

Per-block algorithmic parity is covered by ``test_dit_block.py``. This
test's job is to confirm *assembly* — that the 30-block stack, the
shared timestep modulation, the input / output Linears, and the
weight-loading all wire up correctly.

Slow because the full bf16 → fp32 promotion of ~2.5 GB takes a few
seconds.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from trellis2_mlx.models.dit import SLatFlowConfig, SLatFlowModel
from trellis2_mlx.tests.test_sparse_blocks import _safetensors_load_keys
from trellis2_mlx.utils.weight_convert import slat_flow_model_from_pt_state_dict

pytestmark = [pytest.mark.reference, pytest.mark.slow]

_SLAT_SHAPE_DIT = Path("reference/weights/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")
_SLAT_TEX_DIT = Path("reference/weights/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors")


def _load_full_state(path: Path) -> dict[str, np.ndarray]:
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = __import__("json").loads(f.read(n).decode())
    header.pop("__metadata__", None)
    keys = list(header.keys())
    raw = _safetensors_load_keys(path, keys)
    return {k: v.astype(np.float32) for k, v in raw.items()}


@pytest.fixture(scope="module")
def slat_dit_state() -> dict[str, np.ndarray]:
    if not _SLAT_SHAPE_DIT.exists():
        pytest.skip(f"SLAT shape DiT weights not found at {_SLAT_SHAPE_DIT}")
    return _load_full_state(_SLAT_SHAPE_DIT)


def test_slat_dit_weight_converter_covers_all_keys(slat_dit_state: dict[str, np.ndarray]) -> None:
    """Every PT key must be consumed by our converter (no silently-dropped weights)."""
    pairs = slat_flow_model_from_pt_state_dict(slat_dit_state)
    pt_keys = set(slat_dit_state.keys())
    n_blocks = 30

    consumed: set[str] = set()
    consumed.update(
        ["input_layer.weight", "input_layer.bias", "out_layer.weight", "out_layer.bias"]
    )
    for i in (0, 2):
        consumed.add(f"t_embedder.mlp.{i}.weight")
        consumed.add(f"t_embedder.mlp.{i}.bias")
    consumed.update(["adaLN_modulation.1.weight", "adaLN_modulation.1.bias"])
    for i in range(n_blocks):
        prefix = f"blocks.{i}."
        for k in [
            "modulation",
            "norm2.weight",
            "norm2.bias",
            "self_attn.to_qkv.weight",
            "self_attn.to_qkv.bias",
            "self_attn.to_out.weight",
            "self_attn.to_out.bias",
            "self_attn.q_rms_norm.gamma",
            "self_attn.k_rms_norm.gamma",
            "cross_attn.to_q.weight",
            "cross_attn.to_q.bias",
            "cross_attn.to_kv.weight",
            "cross_attn.to_kv.bias",
            "cross_attn.to_out.weight",
            "cross_attn.to_out.bias",
            "cross_attn.q_rms_norm.gamma",
            "cross_attn.k_rms_norm.gamma",
            "mlp.mlp.0.weight",
            "mlp.mlp.0.bias",
            "mlp.mlp.2.weight",
            "mlp.mlp.2.bias",
        ]:
            consumed.add(prefix + k)
    missing = pt_keys - consumed
    extra = consumed - pt_keys
    assert not missing, f"converter dropped {len(missing)} keys: {sorted(missing)[:5]}"
    assert not extra, f"converter referenced {len(extra)} missing keys: {sorted(extra)[:5]}"
    # 4 top + 4 (t_embedder Linears w+b twice) + 2 (adaLN) + 30 * 21 = 640
    assert len(pairs) == 640


def test_slat_dit_loads_and_runs_smoke(slat_dit_state: dict[str, np.ndarray]) -> None:
    """Load the full 1.3B-param SLAT shape DiT and run one forward pass."""
    cfg = SLatFlowConfig()  # 32 in, 32 out, 30 blocks, 1536 ch, 12 heads
    model = SLatFlowModel(cfg)
    pairs = slat_flow_model_from_pt_state_dict(slat_dit_state)
    model.load_weights(pairs)

    rng = np.random.default_rng(0)
    # Tiny synthetic active set + small context to keep wall-time bounded.
    n_voxels = 16
    n_ctx = 64
    x = mx.array(rng.standard_normal((n_voxels, cfg.in_channels)).astype(np.float32) * 0.5)
    coords = mx.array(rng.integers(0, cfg.resolution, size=(n_voxels, 3), dtype=np.int32))
    t = mx.array([500.0], dtype=mx.float32)  # mid-noise
    cond = mx.array(rng.standard_normal((1, n_ctx, cfg.cond_channels)).astype(np.float32) * 0.5)

    out = model(x, coords, t, cond)
    mx.eval(out)
    out_np = np.asarray(out)

    assert out_np.shape == (n_voxels, cfg.out_channels)
    assert np.isfinite(out_np).all(), "DiT produced non-finite output (NaN / inf)"
    print(
        f"\n  slat-dit smoke OK: 1.3B params, {n_voxels} voxels × {n_ctx} ctx → "
        f"[{n_voxels}, {cfg.out_channels}]  "
        f"out range=[{out_np.min():.3f}, {out_np.max():.3f}]  std={out_np.std():.3f}"
    )


def test_slat_tex_dit_loads_and_runs_smoke() -> None:
    """Same SLatFlowModel class, in_channels=64 (shape latent concatenated as
    extra conditioning). Loads the real texture SLAT DiT and runs one step."""
    if not _SLAT_TEX_DIT.exists():
        pytest.skip(f"SLAT tex DiT weights not found at {_SLAT_TEX_DIT}")
    state = _load_full_state(_SLAT_TEX_DIT)

    cfg = SLatFlowConfig(in_channels=64)  # 32 noise + 32 shape latent
    model = SLatFlowModel(cfg)
    pairs = slat_flow_model_from_pt_state_dict(state)
    model.load_weights(pairs)

    rng = np.random.default_rng(0)
    n_voxels = 16
    n_ctx = 64
    x = mx.array(rng.standard_normal((n_voxels, cfg.in_channels)).astype(np.float32) * 0.5)
    coords = mx.array(rng.integers(0, cfg.resolution, size=(n_voxels, 3), dtype=np.int32))
    t = mx.array([500.0], dtype=mx.float32)
    cond = mx.array(rng.standard_normal((1, n_ctx, cfg.cond_channels)).astype(np.float32) * 0.5)

    out = model(x, coords, t, cond)
    mx.eval(out)
    out_np = np.asarray(out)
    assert out_np.shape == (n_voxels, cfg.out_channels)
    assert np.isfinite(out_np).all()
    print(
        f"\n  slat-tex-dit smoke OK: 1.3B params, in=64 → out=[{n_voxels}, 32]  "
        f"range=[{out_np.min():.3f}, {out_np.max():.3f}]  std={out_np.std():.3f}"
    )


def test_slat_tex_dit_concat_cond_pathway() -> None:
    """Verify the texture-DiT-style ``concat_cond`` forward path: SLatFlowModel
    can receive an extra ``[L, 32]`` shape latent concatenated channel-wise
    onto the ``[L, 32]`` noise to make a ``[L, 64]`` input."""
    if not _SLAT_TEX_DIT.exists():
        pytest.skip(f"SLAT tex DiT weights not found at {_SLAT_TEX_DIT}")
    state = _load_full_state(_SLAT_TEX_DIT)

    cfg = SLatFlowConfig(in_channels=64)
    model = SLatFlowModel(cfg)
    model.load_weights(slat_flow_model_from_pt_state_dict(state))

    rng = np.random.default_rng(0)
    n_voxels = 8
    n_ctx = 32
    # Texture stage's input pattern: noise + shape latent (concat along channel)
    noise = mx.array(rng.standard_normal((n_voxels, 32)).astype(np.float32))
    shape_latent = mx.array(rng.standard_normal((n_voxels, 32)).astype(np.float32))
    coords = mx.array(rng.integers(0, 32, size=(n_voxels, 3), dtype=np.int32))
    t = mx.array([500.0], dtype=mx.float32)
    cond = mx.array(rng.standard_normal((1, n_ctx, 1024)).astype(np.float32))

    out = model(noise, coords, t, cond, concat_cond=shape_latent)
    mx.eval(out)
    assert out.shape == (n_voxels, cfg.out_channels)
    assert np.isfinite(np.asarray(out)).all()
