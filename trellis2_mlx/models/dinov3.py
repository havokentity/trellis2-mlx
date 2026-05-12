"""DINOv3-L image encoder, MLX port.

Implements ``PHASE0_SPEC.md §4.1`` with the corrections recorded in
``docs/open-questions-resolved.md`` Q1 / Q3:

* ViT-L/16 — 24 layers, hidden 1024, 16 heads, patch 16, 4 register tokens.
* 2-D RoPE on the **patch tokens only** (prefix tokens — 1 CLS + 4 registers —
  are not rotated). RoPE uses ``base = rope_theta = 100.0`` against
  patch-centre coordinates normalized to ``[-1, +1]``.
* MLP is the non-gated variant for ViT-L: ``Linear → GELU → Linear`` with
  ``mlp_ratio = 4.0`` (intermediate 4096).
* DINOv3 quirk: the **key projection has no bias** (only Q, V, output do).
* The trellis2 image_feature_extractor uses ``F.layer_norm`` (no learned
  params) on the final hidden states, *not* the model's learned ``self.norm``
  — we follow that.

Reference upstream: ``reference/microsoft-trellis2/trellis2/modules/image_feature_extractor.py``
and ``transformers/models/dinov3_vit/modeling_dinov3_vit.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class DINOv3LConfig:
    """Architectural constants for DINOv3-L/16.

    Defaults match ``facebook/dinov3-vitl16-pretrain-lvd1689m``. Override
    ``image_size`` when running at 1024 for the SLAT-1024 conditioning path.
    """

    image_size: int = 512
    patch_size: int = 16
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    num_register_tokens: int = 4
    rope_theta: float = 100.0
    layerscale_value: float = 1.0
    layer_norm_eps: float = 1e-6
    final_norm_eps: float = 1e-5  # F.layer_norm default in trellis2's extract_features


def _rotate_half(x: mx.array) -> mx.array:
    """Standard RoPE rotate-half: ``[-x_hi, x_lo]``."""
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply 2-D RoPE to ``x`` (over patch tokens only).

    Parameters
    ----------
    x : mx.array
        ``[B, H, N_patches, D]`` query or key tensor.
    cos, sin : mx.array
        ``[1, N_patches, D]`` precomputed rotation phases. The leading
        head dim is broadcast.
    """
    return x * cos[:, None, :, :] + _rotate_half(x) * sin[:, None, :, :]


class PatchEmbeddings(nn.Module):
    """Conv2d patch projection plus CLS + register tokens.

    MLX :class:`mlx.nn.Conv2d` uses NHWC layout — pixel inputs and the
    Conv weight differ from PyTorch by a transpose, handled at load time.
    """

    def __init__(self, cfg: DINOv3LConfig) -> None:
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            in_channels=3,
            out_channels=cfg.hidden_size,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )
        self.cls_token = mx.zeros((1, 1, cfg.hidden_size))
        self.register_tokens = mx.zeros((1, cfg.num_register_tokens, cfg.hidden_size))
        # mask_token is part of the published weights but only used during pre-training; we still
        # carry the parameter so the state dict round-trips cleanly.
        self.mask_token = mx.zeros((1, 1, cfg.hidden_size))

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Embed ``pixel_values`` into a sequence of tokens.

        Parameters
        ----------
        pixel_values : mx.array
            ``[B, H, W, 3]`` image (NHWC, ImageNet-normalized).

        Returns
        -------
        mx.array
            ``[B, 1 + R + N_patches, hidden]`` token sequence, where the
            first ``1 + R`` slots are the CLS and register tokens.
        """
        patches = self.patch_embeddings(pixel_values)  # [B, nH, nW, hidden]
        b, nh, nw, c = patches.shape
        patches = patches.reshape(b, nh * nw, c)
        cls = mx.broadcast_to(self.cls_token, (b, 1, c))
        regs = mx.broadcast_to(self.register_tokens, (b, self.register_tokens.shape[1], c))
        return mx.concatenate([cls, regs, patches], axis=1)


def build_rope_phases(
    cfg: DINOv3LConfig, num_patches_h: int, num_patches_w: int
) -> tuple[mx.array, mx.array]:
    """Compute the 2-D RoPE ``(cos, sin)`` for the patch grid.

    Matches ``DINOv3ViTRopePositionEmbedding.forward``: patch-centre coords
    normalized to ``[-1, +1]``, multiplied by ``inv_freq`` over both axes,
    then flattened/tiled to fill ``head_dim``.
    """
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    # inv_freq: shape (head_dim / 4,) — matches HF exactly
    inv_freq = 1.0 / (cfg.rope_theta ** mx.arange(0, 1, 4 / head_dim, dtype=mx.float32))
    # patch-centre coords in [0, 1] then shifted to [-1, +1]
    coords_h = (mx.arange(num_patches_h, dtype=mx.float32) + 0.5) / num_patches_h
    coords_w = (mx.arange(num_patches_w, dtype=mx.float32) + 0.5) / num_patches_w
    gh, gw = mx.meshgrid(coords_h, coords_w, indexing="ij")
    coords = mx.stack([gh.reshape(-1), gw.reshape(-1)], axis=-1)  # [N, 2]
    coords = 2.0 * coords - 1.0
    # angles: 2π * coord * inv_freq → [N, 2, head_dim/4] → [N, head_dim/2] → [N, head_dim]
    angles = 2.0 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.reshape(num_patches_h * num_patches_w, -1)
    angles = mx.concatenate([angles, angles], axis=-1)
    return mx.cos(angles)[None, :, :], mx.sin(angles)[None, :, :]


class Attention(nn.Module):
    """Multi-head self-attention with 2-D RoPE on patch tokens only.

    DINOv3 uses separate Q/K/V projections (not fused QKV) and **K has no
    bias** (only Q and V do).
    """

    def __init__(self, cfg: DINOv3LConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        num_prefix_tokens: int,
    ) -> mx.array:
        b, n, c = x.shape
        h, d = self.num_heads, self.head_dim
        # Project to Q/K/V and reshape to [B, H, N, D]
        q = self.q_proj(x).reshape(b, n, h, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, n, h, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, n, h, d).transpose(0, 2, 1, 3)
        # RoPE on patch tokens only
        q_pre, q_pat = q[:, :, :num_prefix_tokens], q[:, :, num_prefix_tokens:]
        k_pre, k_pat = k[:, :, :num_prefix_tokens], k[:, :, num_prefix_tokens:]
        q_pat = _apply_rope(q_pat, cos, sin)
        k_pat = _apply_rope(k_pat, cos, sin)
        q = mx.concatenate([q_pre, q_pat], axis=2)
        k = mx.concatenate([k_pre, k_pat], axis=2)
        # SDPA
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, c)
        return self.o_proj(out)


class LayerScale(nn.Module):
    """Per-channel scale applied after the residual sub-layer."""

    def __init__(self, cfg: DINOv3LConfig) -> None:
        super().__init__()
        self.lambda1 = mx.full((cfg.hidden_size,), cfg.layerscale_value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.lambda1


class MLP(nn.Module):
    """Vanilla 2-layer MLP with tanh-approximate GELU (matches transformers default)."""

    def __init__(self, cfg: DINOv3LConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.up_proj(x)))


class TransformerLayer(nn.Module):
    """Pre-norm transformer block with two LayerScales, no drop-path at inference."""

    def __init__(self, cfg: DINOv3LConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attention = Attention(cfg)
        self.layer_scale1 = LayerScale(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)
        self.layer_scale2 = LayerScale(cfg)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        num_prefix_tokens: int,
    ) -> mx.array:
        h = self.attention(self.norm1(x), cos, sin, num_prefix_tokens)
        x = x + self.layer_scale1(h)
        h = self.mlp(self.norm2(x))
        x = x + self.layer_scale2(h)
        return x


class DINOv3L(nn.Module):
    """DINOv3-L/16 image encoder used as the cross-attention KV source.

    The forward pass mirrors ``image_feature_extractor.DinoV3FeatureExtractor.extract_features``
    in the trellis2 upstream: embeddings → RoPE → 24 layers → **parameter-free
    layer norm** over the final hidden state. The model is **frozen** during
    all of training and inference (CLAUDE.md custom-op policy doesn't apply —
    this is a built-in MLX path, no custom Metal kernels).
    """

    def __init__(self, cfg: DINOv3LConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DINOv3LConfig()
        self.embeddings = PatchEmbeddings(self.cfg)
        self.layers = [TransformerLayer(self.cfg) for _ in range(self.cfg.num_hidden_layers)]
        # We keep a learned ``norm`` parameter to match the published checkpoint, even though
        # trellis2's extract_features bypasses it for a parameter-free F.layer_norm.
        self.norm = nn.LayerNorm(self.cfg.hidden_size, eps=self.cfg.layer_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Run the encoder.

        Parameters
        ----------
        pixel_values : mx.array
            ``[B, H, W, 3]`` ImageNet-normalized image batch (NHWC).
            ``H`` and ``W`` must be multiples of ``patch_size``.

        Returns
        -------
        mx.array
            ``[B, 1 + R + N_patches, hidden]`` token sequence after
            parameter-free LayerNorm. Slot 0 is CLS, slots ``1..R`` are
            register tokens, the remainder are spatial patches in row-major
            order.
        """
        if pixel_values.ndim != 4 or pixel_values.shape[-1] != 3:
            raise ValueError(
                f"DINOv3L expects pixel_values shape [B, H, W, 3] (NHWC); "
                f"got {tuple(pixel_values.shape)}"
            )
        _, height, width, _ = pixel_values.shape
        p = self.cfg.patch_size
        if height % p or width % p:
            raise ValueError(f"image size ({height}, {width}) must be divisible by patch_size={p}")
        num_patches_h, num_patches_w = height // p, width // p
        cos, sin = build_rope_phases(self.cfg, num_patches_h, num_patches_w)

        x = self.embeddings(pixel_values)
        num_prefix = 1 + self.cfg.num_register_tokens
        for layer in self.layers:
            x = layer(x, cos, sin, num_prefix)
        # trellis2 uses F.layer_norm without learned params here, not self.norm
        return mx.fast.layer_norm(x, weight=None, bias=None, eps=self.cfg.final_norm_eps)
