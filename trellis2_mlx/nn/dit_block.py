"""DiT block used by all three generator stages.

Implements ``PHASE0_SPEC.md §4.4``. Each block runs:

1. AdaLN-single → modulated self-attention (QK-Norm + RoPE-3D, 12 heads × 128 dim)
2. LayerNorm → cross-attention against DINOv3 image features (no RoPE)
3. AdaLN-single → FFN (1536 → 8192 → 1536; activation pending §8 Q2 verification)

The three stages share this block; they differ only in input projection dim
(stage 3 = 64 because it concatenates the shape latent — see spec §4.4) and
in the conditioning data they receive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    import mlx.core as mx


class DiTBlock(nn.Module):
    """One DiT transformer block (self-attn + cross-attn + FFN), modulated by AdaLN-single."""

    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 12,
        head_dim: int = 128,
        ffn_dim: int = 8192,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        raise NotImplementedError("DiTBlock lands in Phase 1 step 8")

    def __call__(
        self,
        x: "mx.array",
        coords: "mx.array",
        image_kv: "mx.array",
        modulation: "mx.array",
    ) -> "mx.array":
        """Apply the block.

        Parameters
        ----------
        x : mx.array
            ``[L, dim]`` token features.
        coords : mx.array
            ``[L, 3]`` voxel coordinates (for RoPE-3D in self-attention).
        image_kv : mx.array
            ``[Nimg, dim]`` DINOv3 image features (for cross-attention).
        modulation : mx.array
            Six AdaLN-single scalars produced by the shared timestep MLP plus
            this block's per-layer adapter; see spec §4.4 and ``nn/adaln.py``.
        """
        raise NotImplementedError
