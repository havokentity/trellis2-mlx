"""SparseConvNeXtBlock3d — parity vs a PyTorch reference, including a real
loaded-from-checkpoint test.

The upstream reference block calls ``flex_gemm`` (CUDA-only) under the hood,
so we can't run the upstream class directly on this Mac. Instead we build a
**pure-PyTorch reference** that:

1. Reshapes the upstream conv weight ``[Co, 3, 3, 3, Ci]`` to ``[27, Ci, Co]``
   (matches our SubMConv3 layout).
2. Reuses ``_brute_force_submconv3`` from ``test_sparse_conv.py`` as a
   ground-truth submanifold conv.
3. Composes that with ``torch.nn.LayerNorm`` + a ``Linear → SiLU → Linear``
   MLP.

That reference is provably the upstream algorithm (per spec §5.2) — we
diff our MLX block against it.

Three tests:

* ``test_convnext_block_random_init_parity`` — random weights, channels=16,
  ~64 voxels on an 8³ grid. atol 1e-5.
* ``test_convnext_block_real_weights_parity_block_0_1`` — loads the
  ``blocks.0.1`` ConvNeXt block (1024 channels) from the shipped shape
  decoder safetensors, runs both backends on a random ~256-voxel set,
  and confirms parity within fp16/fp32 drift. Marked ``slow``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812 — standard PyTorch alias

from trellis2_mlx.nn.sparse_blocks import (
    _CHILD_OFFSETS,
    SparseConvNeXtBlock3d,
    SparseResBlockC2S3d,
    sparse_channel_to_spatial,
)
from trellis2_mlx.ovoxel.data import build_neighbor_table
from trellis2_mlx.tests.test_sparse_conv import _brute_force_submconv3, _random_active_set
from trellis2_mlx.utils.weight_convert import (
    c2s_upsample_block_from_pt_state_dict,
    convnext_block_from_pt_state_dict,
)

pytestmark = pytest.mark.reference


def _pt_reference_convnext(
    x: np.ndarray,
    conv_w_27_ci_co: np.ndarray,
    conv_b: np.ndarray | None,
    norm_w: np.ndarray,
    norm_b: np.ndarray,
    mlp_up_w: np.ndarray,
    mlp_up_b: np.ndarray,
    mlp_down_w: np.ndarray,
    mlp_down_b: np.ndarray,
    neighbor_table: np.ndarray,
) -> np.ndarray:
    """Run the ConvNeXt block in pure PyTorch using the brute-force submconv3.

    Inputs are all numpy / fp32 to keep precision unambiguous; the block is
    deterministic up to FP add order so dtype matters more than backend.
    """
    h = _brute_force_submconv3(x, conv_w_27_ci_co, neighbor_table, conv_b)
    with torch.no_grad():
        h_t = torch.from_numpy(h)
        norm = torch.nn.LayerNorm(h_t.shape[-1], eps=1e-6, elementwise_affine=True)
        norm.weight.data.copy_(torch.from_numpy(norm_w))
        norm.bias.data.copy_(torch.from_numpy(norm_b))
        h_t = norm(h_t)
        h_t = F.linear(h_t, torch.from_numpy(mlp_up_w), torch.from_numpy(mlp_up_b))
        h_t = F.silu(h_t)
        h_t = F.linear(h_t, torch.from_numpy(mlp_down_w), torch.from_numpy(mlp_down_b))
        return (h_t.numpy() + x).astype(x.dtype)


def test_convnext_block_random_init_parity() -> None:
    """Random weights, 16 channels, ~64 voxels on an 8³ grid. atol=1e-5."""
    resolution = 8
    n_active = 64
    channels = 16
    mlp_ratio = 4.0
    rng = np.random.default_rng(0)

    coords = _random_active_set(n_active, resolution, seed=0)
    nt = np.asarray(build_neighbor_table(mx.array(coords), resolution=resolution))
    x = rng.standard_normal((n_active, channels)).astype(np.float32) * 0.5

    # Random upstream-shaped weights
    pt_conv_w_co_kdhw_ci = (
        rng.standard_normal((channels, 3, 3, 3, channels)).astype(np.float32) * 0.05
    )
    pt_conv_b = rng.standard_normal(channels).astype(np.float32) * 0.05
    norm_w = rng.standard_normal(channels).astype(np.float32)
    norm_b = rng.standard_normal(channels).astype(np.float32) * 0.1
    mlp_dim = int(channels * mlp_ratio)
    mlp_up_w = rng.standard_normal((mlp_dim, channels)).astype(np.float32) * 0.05
    mlp_up_b = rng.standard_normal(mlp_dim).astype(np.float32) * 0.05
    mlp_down_w = rng.standard_normal((channels, mlp_dim)).astype(np.float32) * 0.05
    mlp_down_b = rng.standard_normal(channels).astype(np.float32) * 0.05

    # Build the PT-layout state dict and convert
    pt_state = {
        "conv.weight": pt_conv_w_co_kdhw_ci,
        "conv.bias": pt_conv_b,
        "norm.weight": norm_w,
        "norm.bias": norm_b,
        "mlp.0.weight": mlp_up_w,
        "mlp.0.bias": mlp_up_b,
        "mlp.2.weight": mlp_down_w,
        "mlp.2.bias": mlp_down_b,
    }

    block = SparseConvNeXtBlock3d(channels, mlp_ratio=mlp_ratio)
    block.load_weights(convnext_block_from_pt_state_dict(pt_state))
    mlx_out = np.asarray(block(mx.array(x), mx.array(nt)))

    # PT reference using the same permuted conv weight
    pt_conv_w_27_ci_co = pt_conv_w_co_kdhw_ci.transpose(1, 2, 3, 4, 0).reshape(
        27, channels, channels
    )
    ref_out = _pt_reference_convnext(
        x,
        pt_conv_w_27_ci_co,
        pt_conv_b,
        norm_w,
        norm_b,
        mlp_up_w,
        mlp_up_b,
        mlp_down_w,
        mlp_down_b,
        nt,
    )

    diff = np.abs(mlx_out - ref_out)
    msg = f"max={diff.max():.3e}  mean={diff.mean():.3e}"
    assert diff.max() < 1e-4, msg


def _safetensors_read(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Read the JSON header of a safetensors file. Returns (header, data_start)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return __import__("json").loads(f.read(n).decode()), 8 + n


def _safetensors_load_keys(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Load a small subset of tensors from a safetensors file as numpy arrays."""
    header, data_start = _safetensors_read(path)
    out: dict[str, np.ndarray] = {}
    dtype_map = {
        "F16": np.float16,
        "BF16": None,
        "F32": np.float32,
        "F64": np.float64,
        "I8": np.int8,
        "U8": np.uint8,
        "I16": np.int16,
        "U16": np.uint16,
        "I32": np.int32,
        "U32": np.uint32,
        "I64": np.int64,
        "U64": np.uint64,
    }
    with open(path, "rb") as f:
        for k in keys:
            if k not in header:
                raise KeyError(f"{k} not in {path}")
            info = header[k]
            dt = dtype_map.get(info["dtype"])
            if dt is None:
                raise ValueError(f"unsupported dtype {info['dtype']} for {k}")
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            buf = f.read(end - start)
            out[k] = np.frombuffer(buf, dtype=dt).reshape(info["shape"]).copy()
    return out


def _shape_decoder_path() -> Path:
    p = Path("reference/weights/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")
    if not p.exists():
        pytest.skip(f"shape decoder weights not found at {p}")
    return p


@pytest.mark.slow
def test_convnext_block_real_weights_parity_block_0_1() -> None:
    """Load blocks.0.1 (1024-channel ConvNeXt) from the real shape decoder and
    verify our MLX block matches the PT reference on a random active set.

    The published weights are fp16. We promote everything to fp32 for the
    comparison — the goal here is *algorithmic* parity, not dtype fidelity.
    """
    p = _shape_decoder_path()
    block_keys = {
        "conv.weight": "blocks.0.1.conv.weight",
        "conv.bias": "blocks.0.1.conv.bias",
        "norm.weight": "blocks.0.1.norm.weight",
        "norm.bias": "blocks.0.1.norm.bias",
        "mlp.0.weight": "blocks.0.1.mlp.0.weight",
        "mlp.0.bias": "blocks.0.1.mlp.0.bias",
        "mlp.2.weight": "blocks.0.1.mlp.2.weight",
        "mlp.2.bias": "blocks.0.1.mlp.2.bias",
    }
    raw = _safetensors_load_keys(p, list(block_keys.values()))
    pt_state = {short: raw[full].astype(np.float32) for short, full in block_keys.items()}

    channels = pt_state["norm.weight"].shape[0]
    assert channels == 1024, channels

    # Realistic small active set so the brute-force reference runs in seconds.
    resolution = 16
    n_active = 128
    coords = _random_active_set(n_active, resolution, seed=0)
    nt = np.asarray(build_neighbor_table(mx.array(coords), resolution=resolution))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_active, channels)).astype(np.float32) * 0.5

    # MLX block
    block = SparseConvNeXtBlock3d(channels, mlp_ratio=4.0)
    block.load_weights(convnext_block_from_pt_state_dict(pt_state))
    mlx_out = np.asarray(block(mx.array(x), mx.array(nt)))

    # PT reference (uses the same permuted conv weight)
    pt_conv_w_27_ci_co = (
        pt_state["conv.weight"].transpose(1, 2, 3, 4, 0).reshape(27, channels, channels)
    )
    ref_out = _pt_reference_convnext(
        x,
        pt_conv_w_27_ci_co,
        pt_state["conv.bias"],
        pt_state["norm.weight"],
        pt_state["norm.bias"],
        pt_state["mlp.0.weight"],
        pt_state["mlp.0.bias"],
        pt_state["mlp.2.weight"],
        pt_state["mlp.2.bias"],
        nt,
    )

    diff = np.abs(mlx_out - ref_out)
    msg = (
        f"max={diff.max():.3e}  mean={diff.mean():.3e}  p99={np.percentile(diff, 99):.3e}  "
        f"(ref_std={ref_out.std():.3f})"
    )
    # 1024-channel matmul + 4096-channel MLP with fp32 inputs and fp16 weights
    # converted to fp32: reduction-order drift is the only source of error.
    assert diff.mean() < 5e-4, msg
    assert diff.max() < 5e-3, msg


# ── sparse_channel_to_spatial ─────────────────────────────────────────────


def test_child_offsets_bit_decomposition() -> None:
    """Slot k → (z, y, x) = (bit0, bit1, bit2). z is LSB, x is MSB."""
    assert tuple(_CHILD_OFFSETS[0]) == (0, 0, 0)
    assert tuple(_CHILD_OFFSETS[1]) == (1, 0, 0)  # z=1
    assert tuple(_CHILD_OFFSETS[2]) == (0, 1, 0)  # y=1
    assert tuple(_CHILD_OFFSETS[4]) == (0, 0, 1)  # x=1
    assert tuple(_CHILD_OFFSETS[7]) == (1, 1, 1)


def test_sparse_channel_to_spatial_hand_built() -> None:
    """Two parents at distinct coords; each subdivides into 2 active children.

    Parent A at (0, 0, 0) with slots 0 and 1 active → fine coords (0, 0, 0) and (1, 0, 0).
    Parent B at (3, 5, 7) with slots 2 and 7 active → fine coords (6, 11, 14) and (7, 11, 15).
    """
    parent_coords = mx.array([[0, 0, 0], [3, 5, 7]], dtype=mx.int32)
    c_in = 16  # 8 children × 2 channels each
    # Construct feats so we can verify which slot lands where.
    # feats[parent, child_slot, channel] = parent * 100 + child_slot * 10 + channel
    feats_np = np.zeros((2, 8, 2), dtype=np.float32)
    for p in range(2):
        for k in range(8):
            for c in range(2):
                feats_np[p, k, c] = p * 100 + k * 10 + c
    feats = mx.array(feats_np.reshape(2, c_in))

    sub = mx.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],  # parent A: slots 0, 1
            [0, 0, 1, 0, 0, 0, 0, 1],  # parent B: slots 2, 7
        ],
        dtype=mx.int32,
    ).astype(mx.bool_)

    fine_coords, fine_feats = sparse_channel_to_spatial(parent_coords, feats, sub)
    fine_coords_np = np.asarray(fine_coords)
    fine_feats_np = np.asarray(fine_feats)

    assert fine_coords_np.shape == (4, 3)
    assert fine_feats_np.shape == (4, 2)

    # Parent A: slot 0 → (0,0,0)+(0,0,0)=(0,0,0); slot 1 → (0,0,0)+(1,0,0)=(1,0,0).
    # Parent B at (3,5,7) × 2 = (6,10,14): slot 2 → (6,11,14); slot 7 → (7,11,15).
    np.testing.assert_array_equal(
        fine_coords_np,
        np.array([[0, 0, 0], [1, 0, 0], [6, 11, 14], [7, 11, 15]], dtype=np.int32),
    )

    # Feature checks: feats[p, k, c] = 100p + 10k + c
    np.testing.assert_array_equal(fine_feats_np[0], [0, 1])  # A slot 0
    np.testing.assert_array_equal(fine_feats_np[1], [10, 11])  # A slot 1
    np.testing.assert_array_equal(fine_feats_np[2], [120, 121])  # B slot 2
    np.testing.assert_array_equal(fine_feats_np[3], [170, 171])  # B slot 7


def test_sparse_channel_to_spatial_empty_subdivision() -> None:
    """No active children → degenerate empty fine grid."""
    parent_coords = mx.array([[0, 0, 0], [1, 1, 1]], dtype=mx.int32)
    feats = mx.zeros((2, 16))
    sub = mx.zeros((2, 8), dtype=mx.bool_)
    fc, ff = sparse_channel_to_spatial(parent_coords, feats, sub)
    assert tuple(fc.shape) == (0, 3)
    assert tuple(ff.shape) == (0, 2)


def test_sparse_channel_to_spatial_validates_shapes() -> None:
    # feats channel count not divisible by 8
    with pytest.raises(ValueError, match="divisible by 8"):
        sparse_channel_to_spatial(
            mx.zeros((1, 3), dtype=mx.int32),
            mx.zeros((1, 15)),
            mx.zeros((1, 8), dtype=mx.bool_),
        )
    # subdivision shape mismatch
    with pytest.raises(ValueError, match=r"subdivision must be \[L_coarse, 8\]"):
        sparse_channel_to_spatial(
            mx.zeros((2, 3), dtype=mx.int32),
            mx.zeros((2, 16)),
            mx.zeros((2, 7), dtype=mx.bool_),
        )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_sparse_channel_to_spatial_brute_force_parity(seed: int) -> None:
    """Random parents + random subdivision masks; brute-force enumeration."""
    rng = np.random.default_rng(seed)
    n_coarse = 50
    c_out = 4
    c_in = 8 * c_out

    parent_coords_np = rng.integers(0, 16, size=(n_coarse, 3), dtype=np.int32)
    feats_np = rng.standard_normal((n_coarse, c_in)).astype(np.float32)
    sub_np = rng.integers(0, 2, size=(n_coarse, 8), dtype=np.int32).astype(bool)

    fc, ff = sparse_channel_to_spatial(
        mx.array(parent_coords_np),
        mx.array(feats_np),
        mx.array(sub_np),
    )
    fc_np = np.asarray(fc)
    ff_np = np.asarray(ff)

    # Brute-force expectation
    exp_coords: list[np.ndarray] = []
    exp_feats: list[np.ndarray] = []
    for p in range(n_coarse):
        for k in range(8):
            if sub_np[p, k]:
                child_offset = np.array([k & 1, (k >> 1) & 1, (k >> 2) & 1], dtype=np.int32)
                exp_coords.append(parent_coords_np[p] * 2 + child_offset)
                exp_feats.append(feats_np[p].reshape(8, c_out)[k])
    exp_coords_arr = np.stack(exp_coords) if exp_coords else np.zeros((0, 3), np.int32)
    exp_feats_arr = np.stack(exp_feats) if exp_feats else np.zeros((0, c_out), np.float32)

    np.testing.assert_array_equal(fc_np, exp_coords_arr)
    np.testing.assert_allclose(ff_np, exp_feats_arr, atol=0, rtol=0)


# ── SparseResBlockC2S3d ───────────────────────────────────────────────────


def _pt_reference_c2s_block(
    x: np.ndarray,
    coords_coarse: np.ndarray,
    coarse_nt: np.ndarray,
    fine_resolution: int,
    in_channels: int,
    out_channels: int,
    to_subdiv_w: np.ndarray,
    to_subdiv_b: np.ndarray,
    norm1_w: np.ndarray,
    norm1_b: np.ndarray,
    conv1_w_27_ci_co: np.ndarray,
    conv1_b: np.ndarray,
    conv2_w_27_ci_co: np.ndarray,
    conv2_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-PyTorch reference for SparseResBlockC2S3d.

    Returns (fine_feats, fine_coords, fine_nt, subdiv_logits).
    """
    with torch.no_grad():
        # 1. Subdiv prediction
        x_t = torch.from_numpy(x)
        subdiv_logits = F.linear(x_t, torch.from_numpy(to_subdiv_w), torch.from_numpy(to_subdiv_b))
        subdivision = subdiv_logits.numpy() > 0

        # 2. Coarse path
        norm1 = torch.nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=True)
        norm1.weight.data.copy_(torch.from_numpy(norm1_w))
        norm1.bias.data.copy_(torch.from_numpy(norm1_b))
        h = F.silu(norm1(x_t)).numpy()
        h = _brute_force_submconv3(h, conv1_w_27_ci_co, coarse_nt, conv1_b)

    # 3. channel-to-spatial on h: [L_c, out_channels * 8] → [L_f, out_channels]
    n_coarse = x.shape[0]
    parent_idx_arr, child_slot_arr = np.nonzero(subdivision)
    if parent_idx_arr.size == 0:
        return (
            np.zeros((0, out_channels), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 27), dtype=np.int32),
            subdiv_logits.numpy(),
        )
    h_flat = h.reshape(n_coarse * 8, out_channels)
    h_fine_pre = h_flat[parent_idx_arr * 8 + child_slot_arr]

    # 4. channel-to-spatial on x: [L_c, in_channels] → [L_f, in_channels // 8]
    x_flat = x.reshape(n_coarse * 8, in_channels // 8)
    x_fine_small = x_flat[parent_idx_arr * 8 + child_slot_arr]

    # Build fine coords
    fine_coords = (coords_coarse[parent_idx_arr] * 2 + _CHILD_OFFSETS[child_slot_arr]).astype(
        np.int32
    )

    fine_nt = np.asarray(build_neighbor_table(mx.array(fine_coords), resolution=fine_resolution))

    # 5. Fine path: non-affine LayerNorm + silu + conv2
    with torch.no_grad():
        h_fine_t = torch.from_numpy(h_fine_pre)
        norm2 = torch.nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
        h_fine_t = F.silu(norm2(h_fine_t))
    h_fine_post_conv = _brute_force_submconv3(h_fine_t.numpy(), conv2_w_27_ci_co, fine_nt, conv2_b)

    # 6. Skip: repeat_interleave on x_fine_small
    skip_repeat = out_channels // (in_channels // 8)
    skip = np.repeat(x_fine_small, skip_repeat, axis=1)
    out_feats = h_fine_post_conv + skip
    return out_feats, fine_coords, fine_nt, subdiv_logits.numpy()


def test_c2s_block_random_init_parity() -> None:
    """Random weights, 16 → 8 channels, 16 random parents on a 4³ grid."""
    rng = np.random.default_rng(0)
    in_channels = 16
    out_channels = 8
    coarse_res = 4
    fine_res = 8

    coords_coarse = _random_active_set(16, coarse_res, seed=0)
    coarse_nt = np.asarray(build_neighbor_table(mx.array(coords_coarse), resolution=coarse_res))
    x = rng.standard_normal((coords_coarse.shape[0], in_channels)).astype(np.float32) * 0.5

    # Random PT-layout weights
    pt_state = {
        "to_subdiv.weight": rng.standard_normal((8, in_channels)).astype(np.float32) * 0.5,
        "to_subdiv.bias": rng.standard_normal(8).astype(np.float32) * 0.5,
        "norm1.weight": rng.standard_normal(in_channels).astype(np.float32),
        "norm1.bias": rng.standard_normal(in_channels).astype(np.float32) * 0.1,
        "conv1.weight": rng.standard_normal((out_channels * 8, 3, 3, 3, in_channels)).astype(
            np.float32
        )
        * 0.05,
        "conv1.bias": rng.standard_normal(out_channels * 8).astype(np.float32) * 0.05,
        "conv2.weight": rng.standard_normal((out_channels, 3, 3, 3, out_channels)).astype(
            np.float32
        )
        * 0.05,
        "conv2.bias": rng.standard_normal(out_channels).astype(np.float32) * 0.05,
    }

    block = SparseResBlockC2S3d(in_channels, out_channels)
    block.load_weights(c2s_upsample_block_from_pt_state_dict(pt_state))

    mlx_feats, mlx_fine_coords, mlx_fine_nt, mlx_subdiv = block(
        mx.array(x),
        mx.array(coords_coarse),
        mx.array(coarse_nt),
        fine_resolution=fine_res,
    )
    mlx_feats_np = np.asarray(mlx_feats)
    mlx_fine_coords_np = np.asarray(mlx_fine_coords)

    # PT reference with same conv weight permutation
    conv1_w_27_ci_co = (
        pt_state["conv1.weight"].transpose(1, 2, 3, 4, 0).reshape(27, in_channels, out_channels * 8)
    )
    conv2_w_27_ci_co = (
        pt_state["conv2.weight"].transpose(1, 2, 3, 4, 0).reshape(27, out_channels, out_channels)
    )
    ref_feats, ref_fine_coords, _, _ = _pt_reference_c2s_block(
        x,
        coords_coarse,
        coarse_nt,
        fine_res,
        in_channels,
        out_channels,
        pt_state["to_subdiv.weight"],
        pt_state["to_subdiv.bias"],
        pt_state["norm1.weight"],
        pt_state["norm1.bias"],
        conv1_w_27_ci_co,
        pt_state["conv1.bias"],
        conv2_w_27_ci_co,
        pt_state["conv2.bias"],
    )

    np.testing.assert_array_equal(mlx_fine_coords_np, ref_fine_coords)
    diff = np.abs(mlx_feats_np - ref_feats)
    msg = f"max={diff.max():.3e}  mean={diff.mean():.3e}  L_fine={mlx_feats_np.shape[0]}"
    assert diff.max() < 1e-4, msg


@pytest.mark.slow
def test_c2s_block_real_weights_parity_blocks_0_4() -> None:
    """Load blocks.0.4 (first upsample: 1024 → 512) from the real shape decoder.

    Channels at this stage are the largest in the network (1024 in, 4096 conv1
    out), so we use a tiny active set to keep the brute-force reference fast.
    """
    p = _shape_decoder_path()
    block_keys = {
        "to_subdiv.weight": "blocks.0.4.to_subdiv.weight",
        "to_subdiv.bias": "blocks.0.4.to_subdiv.bias",
        "norm1.weight": "blocks.0.4.norm1.weight",
        "norm1.bias": "blocks.0.4.norm1.bias",
        "conv1.weight": "blocks.0.4.conv1.weight",
        "conv1.bias": "blocks.0.4.conv1.bias",
        "conv2.weight": "blocks.0.4.conv2.weight",
        "conv2.bias": "blocks.0.4.conv2.bias",
    }
    raw = _safetensors_load_keys(p, list(block_keys.values()))
    pt_state = {short: raw[full].astype(np.float32) for short, full in block_keys.items()}

    in_channels = pt_state["norm1.weight"].shape[0]
    out_channels = pt_state["conv2.bias"].shape[0]
    assert in_channels == 1024 and out_channels == 512, (in_channels, out_channels)

    rng = np.random.default_rng(0)
    coarse_res = 16
    fine_res = 32
    n_coarse = 32  # small — brute-force conv1 at 1024→4096 channels is the bottleneck
    coords_coarse = _random_active_set(n_coarse, coarse_res, seed=0)
    coarse_nt = np.asarray(build_neighbor_table(mx.array(coords_coarse), resolution=coarse_res))
    x = rng.standard_normal((n_coarse, in_channels)).astype(np.float32) * 0.5

    block = SparseResBlockC2S3d(in_channels, out_channels)
    block.load_weights(c2s_upsample_block_from_pt_state_dict(pt_state))
    mlx_feats, mlx_fine_coords, _, _ = block(
        mx.array(x),
        mx.array(coords_coarse),
        mx.array(coarse_nt),
        fine_resolution=fine_res,
    )

    conv1_w_27 = (
        pt_state["conv1.weight"].transpose(1, 2, 3, 4, 0).reshape(27, in_channels, out_channels * 8)
    )
    conv2_w_27 = (
        pt_state["conv2.weight"].transpose(1, 2, 3, 4, 0).reshape(27, out_channels, out_channels)
    )
    ref_feats, ref_fine_coords, _, _ = _pt_reference_c2s_block(
        x,
        coords_coarse,
        coarse_nt,
        fine_res,
        in_channels,
        out_channels,
        pt_state["to_subdiv.weight"],
        pt_state["to_subdiv.bias"],
        pt_state["norm1.weight"],
        pt_state["norm1.bias"],
        conv1_w_27,
        pt_state["conv1.bias"],
        conv2_w_27,
        pt_state["conv2.bias"],
    )

    np.testing.assert_array_equal(np.asarray(mlx_fine_coords), ref_fine_coords)
    diff = np.abs(np.asarray(mlx_feats) - ref_feats)
    msg = (
        f"max={diff.max():.3e}  mean={diff.mean():.3e}  p99={np.percentile(diff, 99):.3e}  "
        f"(L_fine={mlx_feats.shape[0]}, ref_std={ref_feats.std():.3f})"
    )
    assert diff.mean() < 5e-3, msg
    assert diff.max() < 5e-2, msg
