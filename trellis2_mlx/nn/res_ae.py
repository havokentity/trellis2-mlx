"""Sparse residual autoencoding blocks (ResEnc / ResDec).

Implements ``PHASE0_SPEC.md §4.2``, §4.3 and §5.4. ResEnc combines a fine→coarse
group-average shortcut with a SubMConv3 + LayerNorm + Linear sequence; ResDec
is its mirror, unstacking 8 children from each parent then ``dup_groups`` to
match the target channel count.

Reference: Chen et al., "Deep Compression Autoencoders" (DC-AE), arXiv 2410.10733.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    import mlx.core as mx


class ResEnc(nn.Module):
    """Down-sampling residual autoencoding block (fine → coarse).

    Steps (per spec §4.2):
    1. Channel-wise group-average over the 8 children of each coarse voxel.
    2. SubMConv3 → LayerNorm → Linear projection of the children's features.
    3. Sum the two paths.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        raise NotImplementedError("ResEnc lands with the VAE in Phase 1 step 6")

    def __call__(
        self,
        x_fine: mx.array,
        coords_fine: mx.array,
        coords_coarse: mx.array,
    ) -> mx.array:
        raise NotImplementedError


class ResDec(nn.Module):
    """Up-sampling residual autoencoding block (coarse → fine).

    Steps (per spec §4.3):
    1. ``unstack``: reshape parent feature into 8 child slots.
    2. ``dup_groups``: tile within each group to reach the target channel count.
    3. Add to the convolutional path.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        raise NotImplementedError

    def __call__(
        self,
        x_coarse: mx.array,
        coords_coarse: mx.array,
        coords_fine: mx.array,
    ) -> mx.array:
        raise NotImplementedError


class EarlyPruningHead(nn.Module):
    """Tiny MLP that predicts an 8-bit child-survival mask per parent voxel.

    See ``PHASE0_SPEC.md §5.5``. At inference the predicted probabilities are
    thresholded (typically 0.5 — exact rule pending §8 Q6) and the surviving
    fine voxels are kept; compaction is handled by the prefix-sum kernel.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        raise NotImplementedError

    def __call__(self, x: mx.array) -> mx.array:
        """Return ``[L_coarse, 8]`` logits/probabilities over child survival."""
        raise NotImplementedError
