"""Trilinear material baking for mesh vertices / texels.

Python wrapper over the Metal kernel implementing ``PHASE0_SPEC.md §5.7``.
For each query point, identifies the 8 surrounding active voxels via the
spatial hash and trilinearly interpolates ``(c, m, r, α)``. Behavior when
fewer than 8 neighbors are present is an open question (spec §8 Q-style
question on trilinear fallback — record in docs/open-questions-resolved.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

    from trellis2_mlx.ovoxel.data import OVoxel


def bake_materials(ovoxel: OVoxel, query_points: mx.array) -> mx.array:
    """Trilinearly sample ``(c, m, r, α)`` at ``query_points``.

    Parameters
    ----------
    ovoxel : OVoxel
        Material-decoded O-Voxel containing per-voxel ``c, m, r, alpha``.
    query_points : mx.array
        Query positions in world coordinates, shape ``[Nq, 3]``.

    Returns
    -------
    mx.array
        Per-query ``(c, m, r, α)`` packed into shape ``[Nq, 6]``.
    """
    raise NotImplementedError
