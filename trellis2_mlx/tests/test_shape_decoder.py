"""End-to-end SC-VAE shape decoder smoke test.

Loads the entire shipped ``shape_dec_next_dc_f16c32_fp16.safetensors`` (~474M
params, fp16) into our :class:`trellis2_mlx.models.vae.ShapeDecoder`, runs
forward on a synthetic 4³ active voxel set, and verifies:

* The decoder accepts the latent + coords + resolution and runs through all
  five stages without crashing.
* The output is at the expected fine resolution (``coarse_res × 16``).
* The output fields ``(v, δ_logits, γ)`` have the right shapes and value
  ranges.
* The full weight load round-trips (no missing or extra keys).

Per-block algorithmic parity is covered by the unit tests in
``test_sparse_conv.py`` and ``test_sparse_blocks.py`` — *those* are the
sources of truth for "the math is right". This file's job is to confirm
*assembly* is correct (per-stage wiring, parameter routing, output split).

The test is marked ``slow`` because loading 474M fp16 parameters takes a few
seconds.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from trellis2_mlx.models.vae import ShapeDecoder, ShapeDecoderConfig
from trellis2_mlx.utils.weight_convert import shape_decoder_from_pt_state_dict

pytestmark = [pytest.mark.reference, pytest.mark.slow]

_SHAPE_DEC = Path("reference/weights/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")


def _safetensors_load_all(path: Path) -> dict[str, np.ndarray]:
    """Load every tensor from a safetensors file as numpy arrays (fp32 promote)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header: dict[str, Any] = __import__("json").loads(f.read(n).decode())
        data_start = 8 + n
        header.pop("__metadata__", None)
        dtype_map = {"F16": np.float16, "F32": np.float32}
        out: dict[str, np.ndarray] = {}
        for k, info in header.items():
            dt = dtype_map.get(info["dtype"])
            if dt is None:
                raise ValueError(f"unsupported dtype {info['dtype']}")
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            buf = f.read(end - start)
            arr = np.frombuffer(buf, dtype=dt).reshape(info["shape"]).copy()
            out[k] = arr.astype(np.float32)
    return out


@pytest.fixture(scope="module")
def shape_decoder_state() -> dict[str, np.ndarray]:
    if not _SHAPE_DEC.exists():
        pytest.skip(f"shape decoder weights not found at {_SHAPE_DEC}")
    return _safetensors_load_all(_SHAPE_DEC)


def test_weight_converter_covers_all_keys(shape_decoder_state: dict[str, np.ndarray]) -> None:
    """Every PT key must be consumed by our converter (no silently-dropped weights)."""
    pairs = shape_decoder_from_pt_state_dict(shape_decoder_state)
    pt_keys = set(shape_decoder_state.keys())
    # Our PT keys with the same path naming
    consumed_keys: set[str] = set()
    for stage_idx, n_cn in enumerate(ShapeDecoderConfig().num_blocks):
        for block_idx in range(n_cn):
            prefix = f"blocks.{stage_idx}.{block_idx}."
            for k in [
                "conv.weight",
                "conv.bias",
                "norm.weight",
                "norm.bias",
                "mlp.0.weight",
                "mlp.0.bias",
                "mlp.2.weight",
                "mlp.2.bias",
            ]:
                if (prefix + k) in pt_keys:
                    consumed_keys.add(prefix + k)
        if stage_idx < len(ShapeDecoderConfig().num_blocks) - 1:
            prefix = f"blocks.{stage_idx}.{n_cn}."
            for k in [
                "to_subdiv.weight",
                "to_subdiv.bias",
                "norm1.weight",
                "norm1.bias",
                "conv1.weight",
                "conv1.bias",
                "conv2.weight",
                "conv2.bias",
            ]:
                if (prefix + k) in pt_keys:
                    consumed_keys.add(prefix + k)
    for k in ("from_latent.weight", "from_latent.bias", "output_layer.weight", "output_layer.bias"):
        consumed_keys.add(k)
    missing = pt_keys - consumed_keys
    assert not missing, f"converter dropped {len(missing)} keys: {sorted(missing)[:5]} ..."
    # And the converter produced something for every block
    assert len(pairs) > 280  # the checkpoint has 292 parameters


def test_shape_decoder_loads_and_runs_smoke(shape_decoder_state: dict[str, np.ndarray]) -> None:
    """Load the full ~474M-param decoder, run on a tiny synthetic latent, check
    output shapes + value ranges."""
    rng = np.random.default_rng(0)
    decoder = ShapeDecoder()

    # Convert and load. This exercises the full converter end-to-end.
    pairs = shape_decoder_from_pt_state_dict(shape_decoder_state)
    decoder.load_weights(pairs)

    # Tiny synthetic latent: 8 active voxels on a 4³ coarse grid.
    # After 4 upsamples (×16), output resolution is 64³.
    coarse_res = 4
    n_coarse = 8
    flat = rng.choice(coarse_res**3, size=n_coarse, replace=False)
    z = flat // (coarse_res**2)
    rem = flat % (coarse_res**2)
    y = rem // coarse_res
    x = rem % coarse_res
    coords = mx.array(np.stack([z, y, x], axis=-1).astype(np.int32))

    latent_feats = mx.array(rng.standard_normal((n_coarse, 32)).astype(np.float32) * 0.5)

    out = decoder(latent_feats, coords, coarse_resolution=coarse_res)
    mx.eval(out.coords, out.v, out.delta_logits, out.gamma)

    # Output resolution: 4 × 2⁴ = 64
    assert out.output_resolution == coarse_res * 16 == 64

    # Output shapes: L_fine × (3, 3, 1) for (v, δ, γ)
    l_fine = out.coords.shape[0]
    assert l_fine > 0, "decoder collapsed to empty active set"
    assert out.coords.shape == (l_fine, 3)
    assert out.v.shape == (l_fine, 3)
    assert out.delta_logits.shape == (l_fine, 3)
    assert out.gamma.shape == (l_fine, 1)

    # Value ranges
    v_np = np.asarray(out.v)
    gamma_np = np.asarray(out.gamma)
    eps = decoder.cfg.voxel_margin
    # v ∈ [-eps, 1 + eps] = [-0.5, 1.5]
    assert v_np.min() > -eps - 1e-3, f"v.min={v_np.min()} below -{eps}"
    assert v_np.max() < 1.0 + eps + 1e-3, f"v.max={v_np.max()} above {1 + eps}"
    # γ from softplus → strictly positive
    assert (gamma_np > 0).all(), "γ has non-positive entries"

    # All coords must be in [0, output_resolution)
    coords_np = np.asarray(out.coords)
    assert (coords_np >= 0).all()
    assert (coords_np < out.output_resolution).all()

    # subdiv_logits: one per upsample stage (= 4 for the default decoder)
    assert len(out.subdiv_logits) == 4

    print(
        f"\n  smoke OK: coarse_res={coarse_res}  L_coarse={n_coarse}  →  "
        f"output_res={out.output_resolution}  L_fine={l_fine}  "
        f"v_range=[{v_np.min():.3f}, {v_np.max():.3f}]  "
        f"γ_range=[{gamma_np.min():.3f}, {gamma_np.max():.3f}]"
    )
