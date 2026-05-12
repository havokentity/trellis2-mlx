"""Sparse-conv-based building blocks for the SC-VAE.

Currently implemented:

* :class:`SparseConvNeXtBlock3d` — the bulk of the VAE decoder (32 instances
  in the published shape decoder). Pattern:
  ``SubMConv3 → LayerNorm(affine) → MLP(Linear → SiLU → Linear) + residual``.

ResEnc / ResDec down/up-sampling blocks and the early-pruning predictor land
once we wire up the full VAE forward pass — see ``PHASE0_SPEC.md §4.2-3``
and ``§5.5``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from trellis2_mlx.nn.sparse_conv import submconv3


class SparseConvNeXtBlock3d(nn.Module):
    """ConvNeXt-style residual block on a sparse voxel set.

    The block holds:

    * ``conv.weight`` — ``[27, C, C]`` SubMConv3 kernel.
    * ``conv.bias`` — ``[C]`` SubMConv3 bias (optional).
    * ``norm.weight`` / ``norm.bias`` — ``[C]`` LayerNorm scale+shift.
    * ``mlp.up.weight`` / ``mlp.up.bias`` — ``[4C, C]``, ``[4C]``.
    * ``mlp.down.weight`` / ``mlp.down.bias`` — ``[C, 4C]``, ``[C]``
      (zero-init in upstream; we follow standard init here and rely on the
      pretrained-weight loader to overwrite it for inference).

    The forward pass mirrors the upstream block (see
    ``reference/microsoft-trellis2/trellis2/models/sc_vaes/sparse_unet_vae.py:284-288``):
    conv → norm → mlp → add residual. The norm is applied **after** the
    conv (post-norm in the ConvNeXt convention) — the original ConvNeXt
    paper applies it inside the residual branch between conv and mlp.

    Parameters
    ----------
    channels : int
        Input/output channel count ``C``. The block is channel-preserving.
    mlp_ratio : float
        Hidden-MLP expansion factor. ``4.0`` for the published checkpoint.
    """

    def __init__(self, channels: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        mlp_dim = int(channels * mlp_ratio)

        # SubMConv3 weight + bias as raw parameters (no nn.Module wrapping) so
        # the converter can write to `conv.weight` / `conv.bias` paths verbatim.
        k_conv = (1.0 / (27 * channels)) ** 0.5
        self.conv_weight = mx.random.uniform(
            low=-k_conv, high=k_conv, shape=(27, channels, channels)
        )
        self.conv_bias = mx.random.uniform(low=-k_conv, high=k_conv, shape=(channels,))

        # LayerNorm with affine scale/shift (matches LayerNorm32 in upstream).
        self.norm = nn.LayerNorm(channels, eps=1e-6, affine=True)

        # MLP: Linear → SiLU → Linear.
        self.mlp_up = nn.Linear(channels, mlp_dim, bias=True)
        self.mlp_down = nn.Linear(mlp_dim, channels, bias=True)

    def __call__(self, x: mx.array, neighbor_table: mx.array) -> mx.array:
        """Apply the block to ``x: [L, C]`` with the active set's neighbor table.

        Returns ``[L, C]`` features. The neighbor table is shared across all
        blocks at the same VAE stage and is built once by the caller via
        :func:`trellis2_mlx.ovoxel.data.build_neighbor_table`.
        """
        h = submconv3(x, self.conv_weight, neighbor_table, self.conv_bias)
        h = self.norm(h)
        h = self.mlp_down(nn.silu(self.mlp_up(h)))
        return h + x
