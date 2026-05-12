"""Submanifold sparse 3D convolution (SubMConv3).

Implements ``PHASE0_SPEC.md §5.2``. The output voxel set equals the input
voxel set (no spatial expansion). Forward computes
``y_i = Σ_k W_k · x_{N(i, k)}`` for valid neighbors ``N(i, k)``; missing
neighbors (encoded ``-1`` in the neighbor table) are masked out.

The forward and backward Metal kernels live in
``trellis2_mlx/metal/kernels/sparse_conv_{fwd,bwd}.metal``; this module is
the MLX ``nn.Module`` wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.nn as nn

if TYPE_CHECKING:
    import mlx.core as mx


class SubMConv3(nn.Module):
    """3×3×3 submanifold sparse convolution.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel counts.
    bias : bool
        Whether to learn an output bias.

    Notes
    -----
    Backward is implemented via a custom Metal kernel so that fine-tuning of
    downstream DiT stages can propagate gradients through the VAE if needed
    (see ``CLAUDE.md`` custom-op policy).
    """

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias
        raise NotImplementedError("SubMConv3 lands in Phase 1 step 5")

    def __call__(self, x: mx.array, neighbor_table: mx.array) -> mx.array:
        """Apply the convolution.

        Parameters
        ----------
        x : mx.array
            ``[L, in_channels]`` voxel features.
        neighbor_table : mx.array
            ``[L, 27]`` int32 table from :func:`trellis2_mlx.ovoxel.data.build_neighbor_table`.
        """
        raise NotImplementedError
