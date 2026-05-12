"""Image preprocessing — RMBG-2.0 background removal and DINOv3 normalization.

Implements ``PHASE0_SPEC.md §2`` step 1. RMBG-2.0 version pinning and the
exact preprocessing transform are open question §8 Q8 — record the answer
in ``docs/open-questions-resolved.md`` once verified.

DINOv3-L expects a fixed input resolution (224 or 518 — see §8 Q3) and its
own normalization statistics; we apply these after compositing the alpha
mask onto a neutral background.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    from PIL.Image import Image


def remove_background(image: Image) -> Image:
    """Run RMBG-2.0 (or equivalent) and return an RGBA image with alpha mask.

    See spec §8 Q8 for the exact upstream version + preprocessing.
    """
    raise NotImplementedError("RMBG wrapper lands in Phase 1 step 3")


def preprocess_for_dinov3(image: Image, *, resolution: int = 224) -> mx.array:
    """Center-crop, resize, and normalize for DINOv3-L input.

    Default resolution is 224; some TRELLIS pipelines use 518 (DINOv2-style) —
    pending verification at §8 Q3.
    """
    raise NotImplementedError
