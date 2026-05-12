"""SC-VAE encoder and decoder for shape and material.

Implements ``PHASE0_SPEC.md §4.2`` (encoder, 354M params) and §4.3 (decoder,
474M params). The decoder mirrors the encoder with three differences:

1. ResEnc → ResDec (unstack + dup_groups instead of group-average).
2. Each upsample is preceded by an early-pruning predictor (spec §5.5).
3. Final projection emits ``(v, δ, γ)`` for shape, ``(c, m, r, α)`` for material.

The material decoder is conditioned on the *shape* decoder's pruning structure
so geometry and material are spatially aligned by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    import mlx.core as mx

    from trellis2_mlx.ovoxel.data import OVoxel


class SCVAEEncoder(nn.Module):
    """Sparse-conv encoder. Not needed for inference (we use Microsoft's
    pretrained checkpoint); included for fine-tuning support."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("SCVAEEncoder lands when fine-tuning support is wired up")


class SCVAEShapeDecoder(nn.Module):
    """Shape decoder producing per-active-voxel ``(v, δ, γ)``.

    Whether ``δ`` is emitted as logits or probabilities is open question
    §8 Q9 — resolve in ``docs/open-questions-resolved.md`` before wiring up
    the loss.
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("SCVAEShapeDecoder lands in Phase 1 step 6")

    def __call__(
        self,
        z_shape: "mx.array",
        active_coords_coarse: "mx.array",
    ) -> "OVoxel":
        """Decode a coarse-grid shape latent into an O-Voxel with shape fields."""
        raise NotImplementedError


class SCVAEMaterialDecoder(nn.Module):
    """Material decoder producing per-active-voxel ``(c, m, r, α)``.

    Conditioned on the shape decoder's pruning structure — same active set at
    every level (spec §4.3).
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

    def __call__(
        self,
        z_material: "mx.array",
        shape_ovoxel: "OVoxel",
    ) -> "OVoxel":
        """Decode material latents into an O-Voxel; returns the input
        ``shape_ovoxel`` augmented with ``c, m, r, α`` (same active set)."""
        raise NotImplementedError
