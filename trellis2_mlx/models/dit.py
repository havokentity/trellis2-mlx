"""The three DiT generator stages.

Implements ``PHASE0_SPEC.md §4.4``. All three share the same block stack
(30 × :class:`trellis2_mlx.nn.dit_block.DiTBlock` with 1536 hidden / 12 × 128
heads) and differ only in:

* Stage 1 — **Sparse-Structure DiT.** Operates on a *dense* ``N³`` latent grid
  (small ``N``, e.g. 32 or 64 — §8 Q5). Predicts binary occupancy.
* Stage 2 — **Geometry DiT.** Operates on the sparse active voxels emitted by
  stage 1. ``in_dim`` = 32.
* Stage 3 — **Material DiT.** Same active set as stage 2; ``in_dim`` = 64
  because it concatenates the geometry latent channel-wise as conditioning.

Each generator targets ~1.3B parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    pass


class DiTStack(nn.Module):
    """Shared backbone for the three DiT stages.

    ``InProj(in_dim → 1536) → 30 × DiTBlock → LayerNorm → OutProj(1536 → 32)``.
    """

    def __init__(self, in_dim: int, depth: int = 30, dim: int = 1536) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.depth = depth
        self.dim = dim
        raise NotImplementedError("DiTStack lands in Phase 1 step 8")


class SparseStructureDiT(DiTStack):
    """Stage 1: predicts which voxels are active on a dense ``N³`` grid."""

    def __init__(self) -> None:
        super().__init__(in_dim=32)


class GeometryDiT(DiTStack):
    """Stage 2: predicts geometry latents ``zᵍ`` on the active voxel set."""

    def __init__(self) -> None:
        super().__init__(in_dim=32)


class MaterialDiT(DiTStack):
    """Stage 3: predicts material latents ``zᵐ`` conditioned on ``zᵍ`` (concat)."""

    def __init__(self) -> None:
        super().__init__(in_dim=64)
