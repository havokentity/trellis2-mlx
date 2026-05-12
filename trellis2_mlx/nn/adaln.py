"""AdaLN-single modulation.

Implements ``PHASE0_SPEC.md §4.4`` and §5.8. A *shared* MLP outside the block
loop maps the timestep embedding to ``(γ₁, β₁, α₁, γ₂, β₂, α₂)``; per-layer
learned adapters then offset those six scalars before they modulate the
attention and FFN sub-blocks. Drastically cheaper than vanilla AdaLN-Zero.

Reference: Chen et al., "PixArt-α", arXiv 2310.00426.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    import mlx.core as mx


class AdaLNSingle(nn.Module):
    """Shared timestep → 6-scalar modulation predictor.

    One instance per DiT model (not per block). Outputs a ``[B, 6 * dim]``
    tensor that each block slices/offsets via its own learned adapter.
    """

    def __init__(self, dim: int = 1536) -> None:
        super().__init__()
        self.dim = dim
        raise NotImplementedError("AdaLNSingle lands with the DiT block in Phase 1 step 8")

    def __call__(self, t_emb: "mx.array") -> "mx.array":
        """Predict shared modulation parameters from timestep embedding ``t_emb``."""
        raise NotImplementedError


def modulate(x: "mx.array", shift: "mx.array", scale: "mx.array") -> "mx.array":
    """Apply ``(1 + scale) * x + shift`` — the AdaLN-single inner op (spec §5.8)."""
    return (1.0 + scale) * x + shift
