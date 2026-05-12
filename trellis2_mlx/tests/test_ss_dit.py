"""Stage 1 (sparse-structure) DiT smoke test.

Same architecture as the SLAT DiTs — 30 blocks × 1536 channels × 12 heads —
but operates on a **dense** 16³ × 8ch grid instead of a sparse active set.
The MLX implementation embeds an :class:`SLatFlowModel` under ``inner`` and
adds a dense ↔ token reshape + a static coord meshgrid.

Two tests:

* ``test_ss_dit_weight_converter_covers_all_keys`` — every key in the real
  ``ss_flow_img_dit_1_3B_64_bf16.safetensors`` is consumed by the converter
  (640 pairs, matching the SLAT DiTs).
* ``test_ss_dit_loads_and_runs_smoke`` — load the full 1.3B-param SS DiT
  and run one forward pass on a synthetic dense latent + DINOv3 context.

Marked ``slow`` (bf16 → fp32 promotion of ~2.5 GB).
"""

from __future__ import annotations

import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from trellis2_mlx.models.dit import SparseStructureFlowConfig, SparseStructureFlowModel
from trellis2_mlx.tests.test_sparse_blocks import _safetensors_load_keys
from trellis2_mlx.utils.weight_convert import ss_flow_model_from_pt_state_dict

pytestmark = [pytest.mark.reference, pytest.mark.slow]

_SS_DIT = Path("reference/weights/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors")


def _load_full_state(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = __import__("json").loads(f.read(n).decode())
    header.pop("__metadata__", None)
    raw = _safetensors_load_keys(path, list(header.keys()))
    return {k: v.astype(np.float32) for k, v in raw.items()}


@pytest.fixture(scope="module")
def ss_dit_state() -> dict[str, np.ndarray]:
    if not _SS_DIT.exists():
        pytest.skip(f"SS DiT weights not found at {_SS_DIT}")
    return _load_full_state(_SS_DIT)


def test_ss_dit_weight_converter_covers_all_keys(ss_dit_state: dict[str, np.ndarray]) -> None:
    """Every PT key must be consumed and routed through ``inner.``."""
    pairs = ss_flow_model_from_pt_state_dict(ss_dit_state)
    assert len(pairs) == 640
    assert all(k.startswith("inner.") for k, _ in pairs)


def test_ss_dit_loads_and_runs_smoke(ss_dit_state: dict[str, np.ndarray]) -> None:
    """Load the full 1.3B-param SS DiT and run a forward pass on a synthetic
    dense latent + DINOv3-shaped context."""
    cfg = SparseStructureFlowConfig()
    model = SparseStructureFlowModel(cfg)
    model.load_weights(ss_flow_model_from_pt_state_dict(ss_dit_state))

    rng = np.random.default_rng(0)
    r = cfg.resolution
    n_ctx = 64
    # Dense [1, C, D, H, W] latent
    x = mx.array(rng.standard_normal((1, cfg.in_channels, r, r, r)).astype(np.float32) * 0.5)
    t = mx.array([500.0], dtype=mx.float32)
    cond = mx.array(rng.standard_normal((1, n_ctx, cfg.cond_channels)).astype(np.float32) * 0.5)

    out = model(x, t, cond)
    mx.eval(out)
    out_np = np.asarray(out)
    expected_shape = (1, cfg.out_channels, r, r, r)
    assert out_np.shape == expected_shape, f"got {out_np.shape} vs {expected_shape}"
    assert np.isfinite(out_np).all()
    print(
        f"\n  ss-dit smoke OK: 1.3B params, dense {r}³ × {cfg.in_channels}ch → "
        f"{r}³ × {cfg.out_channels}ch  range=[{out_np.min():.3f}, {out_np.max():.3f}] "
        f"std={out_np.std():.3f}"
    )
