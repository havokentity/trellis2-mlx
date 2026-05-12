"""Sparse self-attention with QK-Norm and 3D RoPE.

Implements ``PHASE0_SPEC.md §5.1``. Despite operating on tokens from a sparse
voxel grid, the attention itself is *dense* over the active set — at L≈9.6K
there is no sparsity pattern worth exploiting (per spec §5.1 final note).
The Metal kernel is a flash-attention-style tiled implementation with online
softmax.

Q/K are RMS-normalized (QK-Norm, ε=1e-6) and rotated by 3D RoPE *before* the
attention; this module composes those upstream of the Metal kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    import mlx.core as mx


class SparseSelfAttention(nn.Module):
    """Multi-head self-attention with QK-Norm + RoPE-3D, backed by a Metal kernel.

    Parameters
    ----------
    dim : int
        Token feature dim (1536 for the DiT blocks per spec §4.4).
    num_heads : int
        Number of attention heads (12).
    head_dim : int
        Per-head dim (128). ``dim`` must equal ``num_heads * head_dim``.
    rope_base : float
        RoPE base frequency. Default 10000.0; pending verification in
        ``docs/open-questions-resolved.md`` (spec §8 Q1).
    """

    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 12,
        head_dim: int = 128,
        *,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim != num_heads * head_dim:
            raise ValueError(f"dim={dim} must equal num_heads*head_dim={num_heads * head_dim}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        raise NotImplementedError("SparseSelfAttention lands in Phase 1 step 7")

    def __call__(self, x: "mx.array", coords: "mx.array") -> "mx.array":
        """Apply attention.

        Parameters
        ----------
        x : mx.array
            ``[L, dim]`` token features (bf16).
        coords : mx.array
            ``[L, 3]`` voxel coordinates used for RoPE-3D rotation.
        """
        raise NotImplementedError


class CrossAttention(nn.Module):
    """Cross-attention from token stream to DINOv3 image features.

    No RoPE on the image side — image features are unordered (spec §4.4). Q
    is RMS-normalized; K is RMS-normalized; standard scaled dot-product.
    """

    def __init__(self, dim: int = 1536, num_heads: int = 12, head_dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        raise NotImplementedError

    def __call__(self, x: "mx.array", kv: "mx.array") -> "mx.array":
        """Apply cross-attention. ``x`` is ``[L, dim]``, ``kv`` is ``[Nimg, dim]``."""
        raise NotImplementedError
