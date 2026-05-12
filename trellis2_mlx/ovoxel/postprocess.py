"""GLB export from an extracted mesh + baked materials.

Final stage of ``PHASE0_SPEC.md §2`` (step 9). Uses ``trimesh`` to author a
glTF 2.0 / GLB with PBR metallic-roughness channels populated from the baked
material values.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


def export_glb(
    vertices: "mx.array",
    faces: "mx.array",
    materials: "mx.array",
    out_path: str | Path,
) -> Path:
    """Author a GLB file with PBR materials at ``out_path``.

    Parameters
    ----------
    vertices : mx.array
        ``[V, 3]`` mesh vertices.
    faces : mx.array
        ``[F, 3]`` triangle indices.
    materials : mx.array
        ``[V, 6]`` per-vertex ``(c, m, r, α)`` from
        :func:`trellis2_mlx.ovoxel.material_bake.bake_materials`.
    out_path : str | Path
        Destination path (``.glb`` extension expected).
    """
    raise NotImplementedError
