"""Flexible Dual Grid → triangle mesh extraction.

Python wrapper over the Metal kernel implementing ``PHASE0_SPEC.md §5.6``.
For each active voxel, places a dual vertex at ``coord + v``; for each active
edge flag δᵢ, gathers the 4 voxels around that edge into a quad and adaptively
splits the quad into two triangles using γ.

The quad winding-order convention and the exact 4-voxel ring around each edge
are open questions tracked in ``docs/open-questions-resolved.md`` (spec §8 Q7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

    from trellis2_mlx.ovoxel.data import OVoxel


def extract_mesh(ovoxel: OVoxel) -> tuple[mx.array, mx.array]:
    """Return ``(vertices, faces)`` for the input O-Voxel grid.

    Vertices have shape ``[L, 3]`` (one dual vertex per active voxel, in unit
    coords). Faces have shape ``[M, 3]`` where ``M`` is bounded by ``6 * L``
    (3 axes × up to 2 triangles per face). See spec §5.6.
    """
    raise NotImplementedError
