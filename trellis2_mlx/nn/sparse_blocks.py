"""Sparse-conv-based building blocks for the SC-VAE.

Currently implemented:

* :class:`SparseConvNeXtBlock3d` — the bulk of the VAE decoder (32 instances
  in the published shape decoder). Pattern:
  ``SubMConv3 → LayerNorm(affine) → MLP(Linear → SiLU → Linear) + residual``.
* :func:`sparse_channel_to_spatial` — channel→spatial upsample primitive
  used by the decoder's resolution-doubling stages. Reshapes ``[L, C * 8]``
  per-parent features into ``[L_active_children, C]`` per-child features
  given an 8-slot subdivision mask, and emits the corresponding fine-grid
  coordinates.
* :class:`SparseResBlockC2S3d` — the resolution-doubling residual block.
  Predicts per-parent 8-bit subdivision logits, runs SubMConv3 + LayerNorm
  on the coarse grid, reshapes to the fine grid via
  :func:`sparse_channel_to_spatial`, then runs a fine-grid SubMConv3 +
  LayerNorm. Skip connection: ``repeat_interleave(8/factor)`` applied to
  the upsampled-x. Four of these stack to form the 16× spatial upsample
  in the shape / texture decoders.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from trellis2_mlx.nn.sparse_conv import submconv3
from trellis2_mlx.ovoxel.data import build_neighbor_table

# Child-slot index → (z, y, x) bit decomposition used by the upstream
# channel-to-spatial transform: slot 0 = (0,0,0), slot 1 = (1,0,0),
# slot 2 = (0,1,0), …, slot 7 = (1,1,1). z is the least-significant bit,
# x is the most-significant. See
# reference/microsoft-trellis2/trellis2/modules/sparse/spatial/spatial2channel.py:78-83.
_CHILD_OFFSETS = np.array(
    [((slot >> 0) & 1, (slot >> 1) & 1, (slot >> 2) & 1) for slot in range(8)],
    dtype=np.int32,
)  # [8, 3], (z, y, x) bits


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


def sparse_channel_to_spatial(
    coords: mx.array,
    feats: mx.array,
    subdivision: mx.array,
) -> tuple[mx.array, mx.array]:
    """Upsample a sparse voxel grid by 2× via channel→spatial reshape.

    Each parent voxel carries ``C_in = 8 * C_out`` channels — interpreted as
    8 children of ``C_out`` channels each. The ``subdivision`` mask picks
    which children survive; the surviving ones are emitted as fine-grid
    voxels with positions ``2 * parent_coord + child_offset`` (child offsets
    enumerated in :data:`_CHILD_OFFSETS`).

    Mirrors the upstream
    ``trellis2/modules/sparse/spatial/spatial2channel.py:SparseChannel2Spatial``
    (factor=2). Used by ``SparseResBlockC2S3d`` for VAE-decoder resolution
    doubling and by the texture decoder for sub-structure inheritance.

    Parameters
    ----------
    coords : mx.array
        ``[L_coarse, 3]`` int parent voxel coordinates.
    feats : mx.array
        ``[L_coarse, C_in]`` per-parent features. ``C_in`` must be divisible
        by 8.
    subdivision : mx.array
        ``[L_coarse, 8]`` boolean mask. Slot ``k`` is set if child slot ``k``
        of the parent voxel survives the upsample. Child-slot ordering
        follows :data:`_CHILD_OFFSETS` (``z`` is LSB, ``x`` is MSB).

    Returns
    -------
    fine_coords : mx.array
        ``[L_fine, 3]`` int fine voxel coordinates, where
        ``L_fine = subdivision.sum()``. Parent ordering is preserved: all
        children of parent 0 come before any child of parent 1, etc.
    fine_feats : mx.array
        ``[L_fine, C_in // 8]`` per-fine-voxel features.
    """
    if feats.ndim != 2:
        raise ValueError(f"feats must be [L, C], got {tuple(feats.shape)}")
    if feats.shape[1] % 8:
        raise ValueError(f"feats.shape[1] must be divisible by 8, got {feats.shape[1]}")
    if subdivision.shape != (coords.shape[0], 8):
        raise ValueError(
            f"subdivision must be [L_coarse, 8]; got {tuple(subdivision.shape)} vs "
            f"L_coarse={coords.shape[0]}"
        )

    c_out = feats.shape[1] // 8
    # Compaction (active-child enumeration) on numpy — runs once per upsample
    # stage and is dominated by the parent count, not the channel count.
    sub_np = np.asarray(subdivision).astype(bool)
    parent_idx, child_slot = np.nonzero(sub_np)
    # No surviving children at all (degenerate active set) — return empty grid.
    if parent_idx.size == 0:
        return (
            mx.zeros((0, 3), dtype=mx.int32),
            mx.zeros((0, c_out), dtype=feats.dtype),
        )

    # Each fine voxel's coord = 2 * parent_coord + child_offset[child_slot].
    parent_coords_np = np.asarray(coords).astype(np.int32)
    fine_coords_np = parent_coords_np[parent_idx] * 2 + _CHILD_OFFSETS[child_slot]

    # Each fine voxel's feature = feats[parent_idx].reshape(8, C_out)[child_slot]
    # equivalently = feats.reshape(L_c * 8, C_out)[parent_idx * 8 + child_slot]
    flat_idx = mx.array(parent_idx * 8 + child_slot, dtype=mx.int32)
    flat_feats = feats.reshape(-1, c_out)
    fine_feats = mx.take(flat_feats, flat_idx, axis=0)

    return mx.array(fine_coords_np), fine_feats


class SparseResBlockC2S3d(nn.Module):
    """Resolution-doubling residual block (channel-to-spatial variant).

    Implements ``SparseResBlockC2S3d`` from
    ``reference/microsoft-trellis2/trellis2/models/sc_vaes/sparse_unet_vae.py:217-262``.

    Block forward (at one decoder level — coarse → fine):

    1. ``subdiv = to_subdiv(x)`` — predict ``[L_coarse, 8]`` per-parent
       subdivision logits via a single ``Linear``. Threshold at logit 0
       chooses which children survive.
    2. ``h = silu(norm1(x))`` then ``h = conv1(h)`` (SubMConv3) at the
       coarse resolution. Expands channels: ``in_channels → out_channels * 8``.
    3. ``h_fine = channel_to_spatial(h, subdiv)`` — per-parent 8*out
       channels split into 8 children of ``out_channels`` each; surviving
       children become the fine voxel set.
    4. ``x_fine = channel_to_spatial(x, subdiv)`` — same 8-way split of
       the *input* features (``in_channels`` → ``in_channels / 8`` per
       child), giving the residual a coarse-resolution shortcut.
    5. ``h_fine = conv2(silu(norm2(h_fine)))`` at the fine resolution.
       ``norm2`` is non-affine (no learned scale/shift in the checkpoint).
    6. ``out = h_fine + repeat_interleave(x_fine, r, axis=1)`` where
       ``r = out_channels // (in_channels // 8)`` so the skip matches
       ``out_channels``.

    The block carries 8 named parameters (matches the upstream checkpoint
    layout): ``to_subdiv.{weight, bias}``, ``norm1.{weight, bias}``,
    ``conv1.{weight, bias}``, ``conv2.{weight, bias}``. ``norm2`` is
    parameter-free.

    Parameters
    ----------
    in_channels : int
        Coarse-side channel count. Must be divisible by 8.
    out_channels : int
        Fine-side channel count.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels % 8 != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by 8")
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Skip-connection channel-repeat factor — see step 6 of the docstring.
        # After channel_to_spatial(x, subdiv), x is (L_f, in_channels // 8).
        # We need (L_f, out_channels), so repeat each channel `skip_repeat`
        # times along axis 1.
        skip_repeat = out_channels // (in_channels // 8)
        if skip_repeat * (in_channels // 8) != out_channels:
            raise ValueError(
                f"out_channels ({out_channels}) must be a multiple of "
                f"in_channels/8 ({in_channels // 8})"
            )
        self.skip_repeat = skip_repeat

        # Subdivision predictor: Linear(in_channels → 8).
        self.to_subdiv = nn.Linear(in_channels, 8, bias=True)

        # Coarse-side norm + conv (expands channels by 8 to feed the upsample).
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6, affine=True)
        k1 = (1.0 / (27 * in_channels)) ** 0.5
        self.conv1_weight = mx.random.uniform(
            low=-k1, high=k1, shape=(27, in_channels, out_channels * 8)
        )
        self.conv1_bias = mx.random.uniform(low=-k1, high=k1, shape=(out_channels * 8,))

        # Fine-side norm (non-affine) + conv.
        self.norm2 = nn.LayerNorm(out_channels, eps=1e-6, affine=False)
        k2 = (1.0 / (27 * out_channels)) ** 0.5
        self.conv2_weight = mx.random.uniform(
            low=-k2, high=k2, shape=(27, out_channels, out_channels)
        )
        self.conv2_bias = mx.random.uniform(low=-k2, high=k2, shape=(out_channels,))

    def __call__(
        self,
        x: mx.array,
        coords_coarse: mx.array,
        coarse_neighbor_table: mx.array,
        *,
        fine_resolution: int,
        subdivision: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Apply the block.

        Parameters
        ----------
        x : mx.array
            ``[L_coarse, in_channels]`` per-coarse-voxel features.
        coords_coarse : mx.array
            ``[L_coarse, 3]`` int parent coordinates.
        coarse_neighbor_table : mx.array
            ``[L_coarse, 27]`` neighbor table at the coarse grid.
        fine_resolution : int
            Resolution of the fine grid (2× coarse). Needed to build the
            fine-grid neighbor table.
        subdivision : mx.array or None
            Optional ``[L_coarse, 8]`` boolean override of the predicted
            mask. When ``None`` (default), use ``to_subdiv(x) > 0``.

        Returns
        -------
        fine_feats : mx.array
            ``[L_fine, out_channels]`` per-fine-voxel features.
        fine_coords : mx.array
            ``[L_fine, 3]`` int fine-grid coordinates.
        fine_neighbor_table : mx.array
            ``[L_fine, 27]`` neighbor table at the fine grid (built here
            because conv2 needs it; downstream blocks can reuse).
        subdiv_logits : mx.array
            ``[L_coarse, 8]`` raw subdivision logits (for training loss).
        """
        # 1. Predict subdivision (always computed for training; threshold at logit 0).
        subdiv_logits = self.to_subdiv(x)
        if subdivision is None:
            subdivision = subdiv_logits > 0

        # 2. Coarse path: norm1 + silu + conv1 → [L_c, out_channels * 8]
        h = nn.silu(self.norm1(x))
        h = submconv3(h, self.conv1_weight, coarse_neighbor_table, self.conv1_bias)

        # 3. channel_to_spatial of h → [L_f, out_channels]
        fine_coords, h_fine = sparse_channel_to_spatial(coords_coarse, h, subdivision)

        # 4. Skip: channel_to_spatial of x → [L_f, in_channels // 8]
        _, x_fine_small = sparse_channel_to_spatial(coords_coarse, x, subdivision)

        # 5. Fine path: norm2 + silu + conv2 at fine resolution.
        # Build fine neighbor table once and return it for the next stage.
        fine_neighbor_table = build_neighbor_table(fine_coords, resolution=fine_resolution)
        h_fine = nn.silu(self.norm2(h_fine))
        h_fine = submconv3(h_fine, self.conv2_weight, fine_neighbor_table, self.conv2_bias)

        # 6. Skip with channel-repeat → match out_channels.
        skip = mx.repeat(x_fine_small, self.skip_repeat, axis=1)
        out = h_fine + skip
        return out, fine_coords, fine_neighbor_table, subdiv_logits
