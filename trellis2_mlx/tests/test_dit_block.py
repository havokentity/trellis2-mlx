"""DiT block — parity vs a pure-PyTorch reference, including real-weights test.

The upstream block (``ModulatedSparseTransformerCrossBlock``) imports
``flex_gemm`` and a CUDA-only sparse-attention path, so we can't run it on
this Mac. Instead the reference here is a self-contained PyTorch
implementation of the same recipe:

* Self-attn = fused QKV → split → QK-Norm → RoPE-3D → SDPA → output proj
* Cross-attn = ``to_q`` + fused ``to_kv`` → QK-Norm → SDPA → output proj
* Modulation: ``mod = shared_mod + self.modulation``; chunks into
  ``(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)``
* norm1/norm3 non-affine; norm2 affine
* FFN = ``Linear → GELU(tanh) → Linear``

Two tests:

* ``test_dit_block_random_init_parity`` — random weights, channels=192,
  num_heads=6, head_dim=32, 8 voxels. atol ~5e-5.
* ``test_dit_block_real_weights_parity_blocks_0`` — load ``blocks.0`` from
  the real ``slat_flow_img2shape_dit_1_3B_512_bf16.safetensors`` (~32M
  params, bf16), confirm the MLX block matches the PT reference within
  fp16/bf16 drift. Marked ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from trellis2_mlx.nn.dit_block import ModulatedDiTCrossBlock
from trellis2_mlx.tests.test_sparse_blocks import _safetensors_load_keys
from trellis2_mlx.utils.weight_convert import dit_block_from_pt_state_dict

pytestmark = pytest.mark.reference


# ── PyTorch reference ────────────────────────────────────────────────────


def _pt_rope_3d(coords: torch.Tensor, head_dim: int, base: float = 10000.0):
    """Build (cos, sin) [N, head_dim // 2] using upstream's per-axis layout."""

    pairs = head_dim // 2
    freq_dim = pairs // 3
    pad = pairs - 3 * freq_dim
    inv_freq = 1.0 / (base ** (torch.arange(freq_dim, dtype=torch.float32) / freq_dim))
    angles = coords[:, :, None].float() * inv_freq[None, None, :]
    angles = angles.reshape(coords.shape[0], 3 * freq_dim)
    if pad > 0:
        angles = torch.cat([angles, torch.zeros(coords.shape[0], pad)], dim=-1)
    return torch.cos(angles), torch.sin(angles)


def _pt_apply_rope_3d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Complex-pair rotation: (x_real, x_imag) → (real*c - imag*s, real*s + imag*c)."""
    d = x.shape[-1]
    pairs = d // 2
    x_pairs = x.reshape(*x.shape[:-1], pairs, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    while cos.dim() < x_real.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    out = torch.stack([x_real * cos - x_imag * sin, x_real * sin + x_imag * cos], dim=-1)
    return out.reshape(x.shape).to(x.dtype)


def _pt_qk_rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Matches SparseMultiHeadRMSNorm: F.normalize * gamma * sqrt(D)."""
    d = x.shape[-1]
    x32 = x.float()
    out = F.normalize(x32, dim=-1) * gamma * (d**0.5)
    return out.to(x.dtype)


def _pt_self_attn(
    x: torch.Tensor,
    coords: torch.Tensor,
    to_qkv_w: torch.Tensor,
    to_qkv_b: torch.Tensor,
    to_out_w: torch.Tensor,
    to_out_b: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    n, c = x.shape
    h = num_heads
    d = c // h
    qkv = F.linear(x, to_qkv_w, to_qkv_b).reshape(n, 3, h, d)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q = _pt_qk_rms_norm(q, q_gamma)
    k = _pt_qk_rms_norm(k, k_gamma)
    cos, sin = _pt_rope_3d(coords, d)
    q = _pt_apply_rope_3d(q, cos, sin)
    k = _pt_apply_rope_3d(k, cos, sin)
    # [L, H, D] → [1, H, L, D]
    q = q.permute(1, 0, 2).unsqueeze(0)
    k = k.permute(1, 0, 2).unsqueeze(0)
    v = v.permute(1, 0, 2).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.squeeze(0).permute(1, 0, 2).reshape(n, c)
    return F.linear(out, to_out_w, to_out_b)


def _pt_cross_attn(
    x: torch.Tensor,
    context: torch.Tensor,
    to_q_w: torch.Tensor,
    to_q_b: torch.Tensor,
    to_kv_w: torch.Tensor,
    to_kv_b: torch.Tensor,
    to_out_w: torch.Tensor,
    to_out_b: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    n, c = x.shape
    if context.dim() == 3:
        context = context.squeeze(0)
    m, ctx_c = context.shape
    h = num_heads
    d = c // h
    q = F.linear(x, to_q_w, to_q_b).reshape(n, h, d)
    kv = F.linear(context, to_kv_w, to_kv_b).reshape(m, 2, h, d)
    k = kv[:, 0]
    v = kv[:, 1]
    q = _pt_qk_rms_norm(q, q_gamma)
    k = _pt_qk_rms_norm(k, k_gamma)
    q = q.permute(1, 0, 2).unsqueeze(0)
    k = k.permute(1, 0, 2).unsqueeze(0)
    v = v.permute(1, 0, 2).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.squeeze(0).permute(1, 0, 2).reshape(n, c)
    return F.linear(out, to_out_w, to_out_b)


def _pt_dit_block(
    x: torch.Tensor,
    coords: torch.Tensor,
    context: torch.Tensor,
    modulation: torch.Tensor,
    channels: int,
    num_heads: int,
    pt_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Pure-PT reference for one ModulatedDiTCrossBlock forward pass."""
    with torch.no_grad():
        mod = modulation.reshape(-1) + pt_state["modulation"]
        chunks = mod.reshape(6, channels)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_mlp, scale_mlp, gate_mlp = chunks[3], chunks[4], chunks[5]

        # Self-attn
        h = F.layer_norm(x, (channels,), None, None, eps=1e-6)
        h = h * (1.0 + scale_msa) + shift_msa
        h = _pt_self_attn(
            h,
            coords,
            pt_state["self_attn.to_qkv.weight"],
            pt_state["self_attn.to_qkv.bias"],
            pt_state["self_attn.to_out.weight"],
            pt_state["self_attn.to_out.bias"],
            pt_state["self_attn.q_rms_norm.gamma"],
            pt_state["self_attn.k_rms_norm.gamma"],
            num_heads,
        )
        h = h * gate_msa
        x = x + h

        # Cross-attn (norm2 is affine)
        h = F.layer_norm(x, (channels,), pt_state["norm2.weight"], pt_state["norm2.bias"], eps=1e-6)
        h = _pt_cross_attn(
            h,
            context,
            pt_state["cross_attn.to_q.weight"],
            pt_state["cross_attn.to_q.bias"],
            pt_state["cross_attn.to_kv.weight"],
            pt_state["cross_attn.to_kv.bias"],
            pt_state["cross_attn.to_out.weight"],
            pt_state["cross_attn.to_out.bias"],
            pt_state["cross_attn.q_rms_norm.gamma"],
            pt_state["cross_attn.k_rms_norm.gamma"],
            num_heads,
        )
        x = x + h

        # FFN
        h = F.layer_norm(x, (channels,), None, None, eps=1e-6)
        h = h * (1.0 + scale_mlp) + shift_mlp
        h = F.linear(h, pt_state["mlp.mlp.0.weight"], pt_state["mlp.mlp.0.bias"])
        h = F.gelu(h, approximate="tanh")
        h = F.linear(h, pt_state["mlp.mlp.2.weight"], pt_state["mlp.mlp.2.bias"])
        h = h * gate_mlp
        x = x + h
        return x


# ── Tests ────────────────────────────────────────────────────────────────


def _random_state(channels: int, ctx_channels: int, num_heads: int, mlp_ratio: float, seed: int):
    """Build a random PT-layout state dict for a DiT block."""
    rng = np.random.default_rng(seed)
    head_dim = channels // num_heads
    intermediate = int(channels * mlp_ratio)

    def _r(*shape: int) -> torch.Tensor:
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32) * 0.05)

    return {
        "modulation": _r(6 * channels),
        "norm2.weight": torch.from_numpy(rng.standard_normal(channels).astype(np.float32)),
        "norm2.bias": torch.from_numpy(rng.standard_normal(channels).astype(np.float32) * 0.1),
        "self_attn.to_qkv.weight": _r(3 * channels, channels),
        "self_attn.to_qkv.bias": _r(3 * channels),
        "self_attn.to_out.weight": _r(channels, channels),
        "self_attn.to_out.bias": _r(channels),
        "self_attn.q_rms_norm.gamma": torch.ones(num_heads, head_dim)
        + _r(num_heads, head_dim) * 0.1,
        "self_attn.k_rms_norm.gamma": torch.ones(num_heads, head_dim)
        + _r(num_heads, head_dim) * 0.1,
        "cross_attn.to_q.weight": _r(channels, channels),
        "cross_attn.to_q.bias": _r(channels),
        "cross_attn.to_kv.weight": _r(2 * channels, ctx_channels),
        "cross_attn.to_kv.bias": _r(2 * channels),
        "cross_attn.to_out.weight": _r(channels, channels),
        "cross_attn.to_out.bias": _r(channels),
        "cross_attn.q_rms_norm.gamma": torch.ones(num_heads, head_dim)
        + _r(num_heads, head_dim) * 0.1,
        "cross_attn.k_rms_norm.gamma": torch.ones(num_heads, head_dim)
        + _r(num_heads, head_dim) * 0.1,
        "mlp.mlp.0.weight": _r(intermediate, channels),
        "mlp.mlp.0.bias": _r(intermediate),
        "mlp.mlp.2.weight": _r(channels, intermediate),
        "mlp.mlp.2.bias": _r(channels),
    }


def test_dit_block_random_init_parity() -> None:
    """Random weights, channels=192, 6 heads × 32 head_dim. atol 5e-5."""
    channels, num_heads, mlp_ratio = 192, 6, 5.3334
    ctx_channels = 256
    pt_state = _random_state(channels, ctx_channels, num_heads, mlp_ratio, seed=0)

    rng = np.random.default_rng(0)
    n_voxels = 8
    n_ctx = 16
    x = rng.standard_normal((n_voxels, channels)).astype(np.float32) * 0.5
    coords = rng.integers(0, 16, size=(n_voxels, 3), dtype=np.int32)
    context = rng.standard_normal((1, n_ctx, ctx_channels)).astype(np.float32) * 0.5
    shared_mod = rng.standard_normal((6 * channels,)).astype(np.float32) * 0.1

    block = ModulatedDiTCrossBlock(channels, ctx_channels, num_heads, mlp_ratio)
    block.load_weights(dit_block_from_pt_state_dict(pt_state))
    mlx_out = np.asarray(
        block(
            mx.array(x),
            mx.array(coords),
            mx.array(shared_mod),
            mx.array(context),
        )
    )

    ref_out = _pt_dit_block(
        torch.from_numpy(x),
        torch.from_numpy(coords).long(),
        torch.from_numpy(context),
        torch.from_numpy(shared_mod),
        channels,
        num_heads,
        pt_state,
    ).numpy()

    diff = np.abs(mlx_out - ref_out)
    msg = f"max={diff.max():.3e}  mean={diff.mean():.3e}"
    assert diff.max() < 5e-5, msg


@pytest.mark.slow
def test_dit_block_real_weights_parity_blocks_0() -> None:
    """Load real ``blocks.0`` from the shape SLAT DiT (1536 ch, 12 heads).

    Channels are large (8192 FFN intermediate) so we use a small active set
    to keep the brute-force PT reference under a few seconds.
    """
    dit_path = Path("reference/weights/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")
    if not dit_path.exists():
        pytest.skip(f"DiT weights not found at {dit_path}")

    channels, num_heads, mlp_ratio = 1536, 12, 5.3334
    ctx_channels = 1024
    intermediate = int(channels * mlp_ratio)

    block_keys = {
        "modulation": "blocks.0.modulation",
        "norm2.weight": "blocks.0.norm2.weight",
        "norm2.bias": "blocks.0.norm2.bias",
        "self_attn.to_qkv.weight": "blocks.0.self_attn.to_qkv.weight",
        "self_attn.to_qkv.bias": "blocks.0.self_attn.to_qkv.bias",
        "self_attn.to_out.weight": "blocks.0.self_attn.to_out.weight",
        "self_attn.to_out.bias": "blocks.0.self_attn.to_out.bias",
        "self_attn.q_rms_norm.gamma": "blocks.0.self_attn.q_rms_norm.gamma",
        "self_attn.k_rms_norm.gamma": "blocks.0.self_attn.k_rms_norm.gamma",
        "cross_attn.to_q.weight": "blocks.0.cross_attn.to_q.weight",
        "cross_attn.to_q.bias": "blocks.0.cross_attn.to_q.bias",
        "cross_attn.to_kv.weight": "blocks.0.cross_attn.to_kv.weight",
        "cross_attn.to_kv.bias": "blocks.0.cross_attn.to_kv.bias",
        "cross_attn.to_out.weight": "blocks.0.cross_attn.to_out.weight",
        "cross_attn.to_out.bias": "blocks.0.cross_attn.to_out.bias",
        "cross_attn.q_rms_norm.gamma": "blocks.0.cross_attn.q_rms_norm.gamma",
        "cross_attn.k_rms_norm.gamma": "blocks.0.cross_attn.k_rms_norm.gamma",
        "mlp.mlp.0.weight": "blocks.0.mlp.mlp.0.weight",
        "mlp.mlp.0.bias": "blocks.0.mlp.mlp.0.bias",
        "mlp.mlp.2.weight": "blocks.0.mlp.mlp.2.weight",
        "mlp.mlp.2.bias": "blocks.0.mlp.mlp.2.bias",
    }
    raw = _safetensors_load_keys(dit_path, list(block_keys.values()))
    pt_state_np = {short: raw[full].astype(np.float32) for short, full in block_keys.items()}
    pt_state_torch = {k: torch.from_numpy(v) for k, v in pt_state_np.items()}

    # Sanity-check shapes
    assert pt_state_np["self_attn.to_qkv.weight"].shape == (3 * channels, channels)
    assert pt_state_np["mlp.mlp.0.weight"].shape == (intermediate, channels)
    assert pt_state_np["self_attn.q_rms_norm.gamma"].shape == (num_heads, channels // num_heads)

    rng = np.random.default_rng(0)
    n_voxels = 16
    n_ctx = 32
    x = rng.standard_normal((n_voxels, channels)).astype(np.float32) * 0.3
    coords = rng.integers(0, 32, size=(n_voxels, 3), dtype=np.int32)
    context = rng.standard_normal((1, n_ctx, ctx_channels)).astype(np.float32) * 0.3
    shared_mod = rng.standard_normal((6 * channels,)).astype(np.float32) * 0.05

    block = ModulatedDiTCrossBlock(channels, ctx_channels, num_heads, mlp_ratio)
    block.load_weights(dit_block_from_pt_state_dict(pt_state_np))
    mlx_out = np.asarray(
        block(mx.array(x), mx.array(coords), mx.array(shared_mod), mx.array(context))
    )

    ref_out = _pt_dit_block(
        torch.from_numpy(x),
        torch.from_numpy(coords).long(),
        torch.from_numpy(context),
        torch.from_numpy(shared_mod),
        channels,
        num_heads,
        pt_state_torch,
    ).numpy()

    diff = np.abs(mlx_out - ref_out)
    msg = (
        f"max={diff.max():.3e}  mean={diff.mean():.3e}  p99={np.percentile(diff, 99):.3e}  "
        f"(ref_std={ref_out.std():.3f})"
    )
    # 30-layer-equivalent fp32-on-Metal-vs-fp32-on-CPU drift baseline:
    # one layer at 1536 ch with 8192 FFN at fp16 weights → fp32 promotion.
    assert diff.mean() < 5e-4, msg
    assert diff.max() < 5e-3, msg
