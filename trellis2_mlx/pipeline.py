"""End-to-end image-to-3D inference pipeline.

Implements the orchestration described in ``PHASE0_SPEC.md §2`` (steps 1–9):
image preprocessing → DINOv3 encoding → 3-stage DiT diffusion → SC-VAE shape
and material decoders → flexible dual-grid mesh extraction → material baking
→ GLB export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Trellis2Config:
    """Static configuration for a Trellis2 inference run.

    Fields here are *configuration*; runtime tensors live on the pipeline
    instance. The defaults will be filled in once ``PHASE0_SPEC.md §8 Q4–Q5``
    are resolved (sampling step counts, CFG scales, latent grid resolutions).
    """

    output_resolution: int = 1024  # N in {512, 1024, 1536}
    latent_resolution: int | None = None  # filled by config; spec §8 Q5
    dit_steps_structure: int | None = None  # spec §8 Q4
    dit_steps_geometry: int | None = None  # spec §8 Q4
    dit_steps_material: int | None = None  # spec §8 Q4
    cfg_scale_structure: float | None = None  # spec §8 Q4
    cfg_scale_geometry: float | None = None  # spec §8 Q4
    cfg_scale_material: float | None = None  # spec §8 Q4


class Trellis2ImageTo3DPipeline:
    """Top-level image-to-3D pipeline. See ``PHASE0_SPEC.md §2``.

    The pipeline is constructed once (loading DINOv3, the three DiTs, and
    both SC-VAE decoders) and called per image. The shape and material
    decoders share spatial structure by construction — the material decoder
    is conditioned on the shape decoder's pruning decisions (see spec §4.3).
    """

    def __init__(self, config: Trellis2Config, weights_dir: str | Path) -> None:
        self.config = config
        self.weights_dir = Path(weights_dir)
        raise NotImplementedError("pipeline construction lands in Phase 1 step 8+")

    def __call__(self, image: Any, seed: int | None = None) -> dict[str, Any]:
        """Run the full image-to-3D pipeline. Returns ``{'mesh': ..., 'materials': ...}``."""
        raise NotImplementedError
