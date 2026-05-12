"""DINOv3-L image encoder (frozen).

Implements ``PHASE0_SPEC.md §4.1``. ViT-L/16: 24 layers, 1024 dim, 16 heads,
16×16 patch size. Input resolution and which-layer-features pending §8 Q3.

This is a vanilla ViT — no custom Metal kernels needed. The optimization
opportunity (per spec §4.1) is converting to CoreML and running on ANE so the
GPU stays free for the DiT/VAE stages; that's a Phase 2+ concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    import mlx.core as mx


class DINOv3L(nn.Module):
    """ViT-L/16 image encoder used to condition all three DiT stages.

    Frozen during all training and inference. Loaded from
    ``facebook/dinov3-vitl16-pretrain-lvd1689m`` weights.
    """

    def __init__(
        self,
        image_size: int = 224,  # spec §8 Q3 — may be 518 like DINOv2
        patch_size: int = 16,
        dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        raise NotImplementedError("DINOv3L lands in Phase 1 step 3")

    def __call__(self, image: "mx.array") -> "mx.array":
        """Return per-patch tokens (and possibly CLS) used as cross-attn KV."""
        raise NotImplementedError
