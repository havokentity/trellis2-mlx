"""SS-VAE decoder — load real weights and confirm 16³ → 64³ output.

The SS-VAE decoder is the bridge between the stage-1 sparse-structure DiT
output (a 16³ × 8ch dense latent) and the active voxel set that drives
stages 2/3. Forward pass: ``[1, 8, 16, 16, 16]`` → ``[1, 1, 64, 64, 64]``
occupancy logits.

Weights are from ``microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16``
(a separate HF repo from TRELLIS.2-4B). The pipeline.json in TRELLIS.2-4B
references this exact path, so it's a hard dependency.

Two tests:

* ``test_ss_decoder_weight_converter_covers_all_keys`` — every PT key
  consumed; 74 pairs.
* ``test_ss_decoder_loads_and_runs_smoke`` — real weights, one forward
  pass on a synthetic 16³ × 8ch latent. Verifies the output shape, that
  values are finite, and the implied 4× spatial upsample (16 → 64).
"""

from __future__ import annotations

import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from trellis2_mlx.models.vae import SparseStructureDecoder, SparseStructureDecoderConfig
from trellis2_mlx.tests.test_sparse_blocks import _safetensors_load_keys
from trellis2_mlx.utils.weight_convert import ss_decoder_from_pt_state_dict

pytestmark = [pytest.mark.reference, pytest.mark.slow]

_SS_DECODER = Path("reference/weights/trellis-1/ss_dec_conv3d_16l8_fp16.safetensors")


def _load_full_state(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = __import__("json").loads(f.read(n).decode())
    header.pop("__metadata__", None)
    raw = _safetensors_load_keys(path, list(header.keys()))
    return {k: v.astype(np.float32) for k, v in raw.items()}


@pytest.fixture(scope="module")
def ss_decoder_state() -> dict[str, np.ndarray]:
    if not _SS_DECODER.exists():
        pytest.skip(
            f"SS decoder weights not found at {_SS_DECODER} — download with "
            "hf_hub_download('microsoft/TRELLIS-image-large', 'ckpts/ss_dec_conv3d_16l8_fp16.safetensors')"
        )
    return _load_full_state(_SS_DECODER)


def test_ss_decoder_weight_converter_covers_all_keys(
    ss_decoder_state: dict[str, np.ndarray],
) -> None:
    """Every PT key must be consumed. Checkpoint has 74 params."""
    pairs = ss_decoder_from_pt_state_dict(ss_decoder_state)
    assert len(pairs) == len(ss_decoder_state)


def test_ss_decoder_loads_and_runs_smoke(
    ss_decoder_state: dict[str, np.ndarray],
) -> None:
    """Load the full SS decoder and run a forward pass on synthetic input.

    Confirms the dense Conv3d path works on Metal and the channel/spatial
    layout is correct end-to-end (the NCDHW ↔ NDHWC weight conversion and
    the pixel-shuffle upsample are the riskiest pieces).
    """
    cfg = SparseStructureDecoderConfig()
    model = SparseStructureDecoder(cfg)
    model.load_weights(ss_decoder_from_pt_state_dict(ss_decoder_state))

    rng = np.random.default_rng(0)
    # PT-style NCDHW input — the model auto-detects layout.
    z = mx.array(rng.standard_normal((1, cfg.latent_channels, 16, 16, 16)).astype(np.float32) * 0.5)
    out = model(z)
    mx.eval(out)
    out_np = np.asarray(out)

    # Output should be NCDHW (matches input convention) with spatial = 4 × 16 = 64
    # (2 upsample stages × ×2 each) and 1 channel.
    assert out_np.shape == (1, cfg.out_channels, 64, 64, 64), (
        f"got {out_np.shape}, expected (1, 1, 64, 64, 64)"
    )
    assert np.isfinite(out_np).all()

    # Count positives — the SS decoder is trained so the >0 threshold gives
    # the active set. For a random latent we expect a roughly half-and-half
    # split (the field hasn't been pushed strongly in either direction).
    positives = float((out_np > 0).mean())
    print(
        f"\n  ss-decoder smoke OK: 16³ × 8ch → 64³ × 1ch  "
        f"range=[{out_np.min():.3f}, {out_np.max():.3f}]  "
        f">0 fraction={positives:.3f}"
    )
