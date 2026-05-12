"""O-Voxel container and auxiliary index structures.

Implements ``PHASE0_SPEC.md §3.2`` (Structure-of-Arrays layout) and the
auxiliary structures from §3.3 (spatial hash, neighbor table, child→parent
map, pruning mask). The neighbor table is the hot path for sparse conv —
the actual GPU-side construction kernel is in
``trellis2_mlx/metal/kernels/neighbor_build.metal``.

We use ``int32`` for coordinates rather than ``uint16`` because MLX has no
native ``uint16`` (see spec §3.2 and §6.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


@dataclass
class OVoxel:
    """Per-active-voxel SoA container. See ``PHASE0_SPEC.md §3.1``.

    Attributes
    ----------
    coords : mx.array
        Integer voxel coordinates, shape ``[L, 3]``, dtype ``int32``.
    v : mx.array
        Dual vertex offset within each voxel, shape ``[L, 3]``, range ``[0, 1]``.
    delta : mx.array
        Active-edge flags on the −X/−Y/−Z faces, shape ``[L, 3]``.
        Held as float during differentiable training; cast to uint at export.
    gamma : mx.array
        Quad-split weight, shape ``[L]``, range ``(0, 1)``.
    c : mx.array
        Base color (linear RGB), shape ``[L, 3]``.
    m : mx.array
        Metallic, shape ``[L]``.
    r : mx.array
        Roughness, shape ``[L]``.
    alpha : mx.array
        Opacity, shape ``[L]``.
    resolution : int
        Grid resolution ``N`` (typically 512 / 1024 / 1536).
    """

    coords: mx.array
    v: mx.array
    delta: mx.array
    gamma: mx.array
    c: mx.array
    m: mx.array
    r: mx.array
    alpha: mx.array
    resolution: int


def build_spatial_hash(coords: mx.array) -> mx.array:
    """Construct a parallel open-addressing hash table over voxel coords.

    See ``PHASE0_SPEC.md §3.3`` and ``§5.3``. The GPU implementation lives in
    ``trellis2_mlx/metal/kernels/neighbor_build.metal``; this function is the
    Python-side dispatcher.
    """
    raise NotImplementedError


def build_neighbor_table(coords: mx.array) -> mx.array:
    """Build the ``[L, 27]`` int32 neighbor table for submanifold sparse conv.

    See ``PHASE0_SPEC.md §3.3``. Each row of the output table maps a voxel to
    its 27 neighbors in z-y-x scan order; missing neighbors are encoded as -1.
    Implementation calls the Metal kernel via ``trellis2_mlx.metal.ops``.
    """
    raise NotImplementedError


def child_to_parent(coords: mx.array) -> mx.array:
    """Map fine voxels to coarse parents (used by SC-VAE down/up). See spec §3.3."""
    raise NotImplementedError
