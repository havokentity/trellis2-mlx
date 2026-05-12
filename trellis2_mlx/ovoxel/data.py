"""O-Voxel container and auxiliary index structures.

Implements ``PHASE0_SPEC.md §3.2`` (Structure-of-Arrays layout) and the
auxiliary structures from §3.3 (neighbor table and child→parent map).

The 1024³ active set is ~9.6K voxels; the neighbor table is ``[L, 27] int32``
≈ 1 MB and fits comfortably in M4 L2. The current implementation is
**numpy-backed** — fully correct against the spec, but runs on CPU. The
Metal kernel that replaces the inner hashing loop is in
``trellis2_mlx/metal/kernels/neighbor_build.metal`` (Phase 1 step 4 in
``PHASE0_SPEC.md §9``); it preserves the API in this module so callers
don't change.

We use ``int32`` for coordinates rather than ``uint16`` because MLX has no
native ``uint16`` (see spec §3.2 and §6.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# 27 neighbor offsets in z-y-x scan order: (dz, dy, dx) ∈ {-1, 0, +1}³.
# Slot 13 = (0, 0, 0) is the centre voxel (self-neighbor).
_NEIGHBOR_OFFSETS = np.array(
    [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
    dtype=np.int32,
)  # [27, 3]


@dataclass
class OVoxel:
    """Per-active-voxel SoA container. See ``PHASE0_SPEC.md §3.1``.

    Attributes
    ----------
    coords : mx.array
        Integer voxel coordinates, shape ``[L, 3]``, dtype ``int32``.
        Coordinate order is ``(z, y, x)`` matching the spec's scan order.
    v : mx.array
        Dual vertex offset within each voxel, shape ``[L, 3]``. Range is
        ``[-voxel_margin, 1 + voxel_margin]`` (default margin 0.5 → ``[-0.5, 1.5]``)
        — see ``docs/open-questions-resolved.md`` Q9.
    delta : mx.array
        Active-edge flags on the −X/−Y/−Z faces, shape ``[L, 3]``.
        Decoder emits raw logits; threshold at 0 for inference.
    gamma : mx.array
        Per-voxel quad-split weight, shape ``[L]``. Softplus-mapped so
        range is ``(0, ∞)``. The mesh extractor compares
        ``γ[0]·γ[2]`` vs ``γ[1]·γ[3]`` along each quad's diagonals.
    c : mx.array
        Base color (linear RGB), shape ``[L, 3]``.
    m : mx.array
        Metallic, shape ``[L]``.
    r : mx.array
        Roughness, shape ``[L]``.
    alpha : mx.array
        Opacity, shape ``[L]``.
    resolution : int
        Grid resolution ``N`` (typically 32 / 64 at the SLAT bottleneck or
        512 / 1024 / 1536 at the SC-VAE output).
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

    @property
    def num_active(self) -> int:
        """Number of active voxels ``L``."""
        return int(self.coords.shape[0])


def _coords_to_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    """Encode ``[L, 3]`` int coords as ``[L]`` int64 linear keys.

    ``key = z * N² + y * N + x`` — at 1024³ this is ≤ 2³⁰ so int64 is overkill
    for storage but matches NumPy's default ``searchsorted`` type and avoids
    silent overflow if the grid is ever scaled up.
    """
    z = coords[:, 0].astype(np.int64)
    y = coords[:, 1].astype(np.int64)
    x = coords[:, 2].astype(np.int64)
    return z * (resolution * resolution) + y * resolution + x


def build_neighbor_table(coords: mx.array, *, resolution: int) -> mx.array:
    """Build the ``[L, 27]`` int32 neighbor table for submanifold sparse conv.

    Implements ``PHASE0_SPEC.md §3.3`` / §5.3. For each active voxel ``i``,
    row ``i`` contains the indices of the 27 surrounding voxels in z-y-x scan
    order (slot 13 is the voxel itself), or ``-1`` if a neighbor is not in
    the active set or lies outside the ``[0, N)³`` grid.

    Parameters
    ----------
    coords : mx.array
        ``[L, 3]`` int voxel coordinates. Order ``(z, y, x)``.
    resolution : int
        Grid resolution ``N``. Required to bounds-check neighbors and to
        encode coords as linear keys.

    Returns
    -------
    mx.array
        ``[L, 27]`` int32 neighbor index table.
    """
    coords_np = np.asarray(coords)
    if coords_np.ndim != 2 or coords_np.shape[1] != 3:
        raise ValueError(f"coords must be shape [L, 3], got {coords_np.shape}")
    if coords_np.size == 0:
        return mx.zeros((0, 27), dtype=mx.int32)

    n_active = coords_np.shape[0]
    n = resolution

    if (coords_np < 0).any() or (coords_np >= n).any():
        raise ValueError(
            f"coords contain values outside [0, {n}); min={coords_np.min()} max={coords_np.max()}"
        )

    # 1-D linear keys for active voxels, sorted (so we can binary-search).
    active_keys = _coords_to_keys(coords_np, n)
    sort_idx = np.argsort(active_keys, kind="stable")
    sorted_keys = active_keys[sort_idx]

    # All neighbor coordinates: [L, 27, 3]
    neighbor_coords = coords_np[:, None, :].astype(np.int64) + _NEIGHBOR_OFFSETS[None, :, :]
    in_bounds = ((neighbor_coords >= 0) & (neighbor_coords < n)).all(axis=-1)  # [L, 27]

    # Clamp before key encoding so the searchsorted call gets a valid range
    # even for out-of-bounds slots; the in_bounds mask zeros them out below.
    clamped = np.clip(neighbor_coords, 0, n - 1)
    neighbor_keys = _coords_to_keys(clamped.reshape(-1, 3), n).reshape(n_active, 27)

    flat_neighbor_keys = neighbor_keys.reshape(-1)
    positions = np.searchsorted(sorted_keys, flat_neighbor_keys)
    positions = np.clip(positions, 0, n_active - 1)
    matched_keys = sorted_keys[positions]
    found = matched_keys == flat_neighbor_keys

    original_indices = sort_idx[positions]
    result = np.where(found & in_bounds.reshape(-1), original_indices, -1).astype(np.int32)
    return mx.array(result.reshape(n_active, 27))


def child_to_parent_coords(coords: mx.array) -> mx.array:
    """Return the coarse parent coordinate for each fine voxel.

    Implements ``PHASE0_SPEC.md §3.3``: ``parent_coord = coord >> 1``. The
    spec's 8-children-to-1-parent mapping is used by the SC-VAE down/up
    stages; this helper returns *only the parent coords* — the grouping
    (which children belong to which parent) is recovered by sorting on the
    returned coords.

    Parameters
    ----------
    coords : mx.array
        ``[L_fine, 3]`` fine-grid coords.

    Returns
    -------
    mx.array
        ``[L_fine, 3]`` int parent coords on the half-resolution grid.
    """
    return (coords.astype(mx.int32)) // 2


def neighbor_offset_index(dz: int, dy: int, dx: int) -> int:
    """Convert a ``(dz, dy, dx) ∈ {-1, 0, +1}³`` offset to its scan-order slot.

    Slot 0 is ``(-1, -1, -1)``; slot 13 is ``(0, 0, 0)`` (self); slot 26 is
    ``(+1, +1, +1)``. Useful when consuming the neighbor table by axis (e.g.
    a SubMConv3 kernel iterating ``k=0..26``).
    """
    if not (-1 <= dz <= 1 and -1 <= dy <= 1 and -1 <= dx <= 1):
        raise ValueError(f"offsets must be in {{-1, 0, 1}}; got ({dz}, {dy}, {dx})")
    return (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)
