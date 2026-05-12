"""Sparse-token self- and cross-attention for the DiT generators.

Implements ``PHASE0_SPEC.md §5.1`` (sparse attention). Uses MLX's built-in
:func:`mx.fast.scaled_dot_product_attention` until the custom Metal flash
kernel lands (Phase 1 step 7). The functional pieces:

* :class:`SparseMultiHeadRMSNorm` — per-head, per-channel RMSNorm matching
  ``trellis2/modules/sparse/attention/modules.py:11-24``. Note the upstream
  uses ``F.normalize`` + multiplies by ``sqrt(D)`` rather than the standard
  ``RMSNorm`` formulation; mathematically equivalent for ``eps → 0`` but
  worth preserving so the gamma values from the checkpoint mean the same
  thing.
* :class:`SparseSelfAttention` — fused ``to_qkv`` + optional QK-Norm +
  optional RoPE-3D + SDPA + ``to_out``.
* :class:`SparseCrossAttention` — split ``to_q`` and fused ``to_kv``,
  optional QK-Norm, no RoPE (image-feature context is unordered).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from trellis2_mlx.nn.rope import apply_rope_3d, build_rope_3d_phases


class SparseMultiHeadRMSNorm(nn.Module):
    """Per-head, per-channel RMSNorm for Q and K.

    Upstream uses ``F.normalize(x, dim=-1) * gamma * sqrt(D)`` rather than
    the more usual ``x / RMS * gamma`` — both compute
    ``x / ||x|| * gamma * sqrt(D)`` modulo eps. We mirror the upstream
    formula so the published ``gamma`` values mean the same thing.
    """

    def __init__(self, head_dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = head_dim**0.5
        self.gamma = mx.ones((num_heads, head_dim))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply per-head RMSNorm to ``x: [..., H, D]``.

        Computation runs in fp32 for stability — matches the upstream
        ``x.float()`` cast at the top of ``SparseMultiHeadRMSNorm.forward``.
        """
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)
        # L2 norm along the head_dim axis; broadcast-compatible with gamma [H, D].
        norm = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True)) + 1e-12
        out = (x32 / norm) * self.gamma * self.scale
        return out.astype(orig_dtype)


class SparseSelfAttention(nn.Module):
    """Multi-head self-attention with QK-Norm + RoPE-3D on sparse tokens.

    Fused QKV projection (matches the checkpoint layout
    ``to_qkv.weight: [3*C, C]``). Both QK-Norm and RoPE-3D are enabled for
    every block of every published DiT (``qk_rms_norm: true``, ``pe_mode: rope``).

    Parameters
    ----------
    channels : int
    num_heads : int
    rope_base : float
        Default 10000.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        *,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.rope_base = rope_base

        self.to_qkv = nn.Linear(channels, 3 * channels, bias=True)
        self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels, bias=True)

    def __call__(self, x: mx.array, coords: mx.array) -> mx.array:
        """Apply self-attention.

        Parameters
        ----------
        x : mx.array
            ``[L, channels]`` per-voxel features.
        coords : mx.array
            ``[L, 3]`` voxel coordinates for RoPE-3D rotation.
        """
        n, _ = x.shape
        h, d = self.num_heads, self.head_dim
        # Fused QKV projection then split.
        qkv = self.to_qkv(x).reshape(n, 3, h, d)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [L, H, D]
        # QK-Norm
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
        # RoPE-3D on Q and K (V is not rotated)
        cos, sin = build_rope_3d_phases(coords, d, base=self.rope_base)
        q = apply_rope_3d(q, cos, sin)
        k = apply_rope_3d(k, cos, sin)
        # SDPA expects [B, H, L, D]; we have [L, H, D] — add a batch dim of 1.
        q = q.transpose(1, 0, 2)[None]  # [1, H, L, D]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        scale = d**-0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out[0].transpose(1, 0, 2).reshape(n, h * d)
        return self.to_out(out)


class SparseCrossAttention(nn.Module):
    """Cross-attention from sparse Q to a dense image-feature KV.

    No RoPE on the cross-attention side — image features are unordered.
    QK-Norm is enabled in the published checkpoint (``qk_rms_norm_cross: true``).

    Parameters
    ----------
    channels : int
        Sparse-side hidden dim (Q comes from a ``[L, channels]`` tensor).
    ctx_channels : int
        Context-side hidden dim (K, V come from a ``[B, M, ctx_channels]``
        tensor — typically 1024 for the DINOv3-L features).
    num_heads : int
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.ctx_channels = ctx_channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_kv = nn.Linear(ctx_channels, 2 * channels, bias=True)
        self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels, bias=True)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        """Apply cross-attention.

        Parameters
        ----------
        x : mx.array
            ``[L, channels]`` per-voxel query features.
        context : mx.array
            ``[B, M, ctx_channels]`` (batched) or ``[M, ctx_channels]``
            (single-batch) image feature tokens. For B > 1 the caller is
            responsible for batching voxels too (this codebase currently
            assumes B = 1).
        """
        n, _ = x.shape
        h, d = self.num_heads, self.head_dim
        if context.ndim == 3:
            if context.shape[0] != 1:
                raise NotImplementedError(
                    f"batched cross-attn (B={context.shape[0]}) not yet supported; "
                    "the SC-VAE / DiT pipeline runs at B=1 in inference"
                )
            context = context[0]
        # Q from sparse, K/V from context
        q = self.to_q(x).reshape(n, h, d)
        kv = self.to_kv(context).reshape(context.shape[0], 2, h, d)
        k = kv[:, 0]  # [M, H, D]
        v = kv[:, 1]
        # QK-Norm
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
        # SDPA: Q has L tokens, K/V have M tokens. Reshape to [1, H, L, D] / [1, H, M, D].
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        scale = d**-0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out[0].transpose(1, 0, 2).reshape(n, h * d)
        return self.to_out(out)


__all__ = ["SparseCrossAttention", "SparseMultiHeadRMSNorm", "SparseSelfAttention"]
