"""GLB export from an extracted mesh.

Final stage of ``PHASE0_SPEC.md §2`` (step 9). Uses ``trimesh`` to author a
glTF 2.0 / GLB. Materials are deliberately not authored here — that lands
once the material decoder is in (per-vertex base color / metallic / roughness
go into PBR slots). For now we export a plain triangulated mesh which is
enough to inspect the geometry in any 3D viewer.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import trimesh


def export_glb(
    vertices: mx.array,
    faces: mx.array,
    out_path: str | Path,
    *,
    material_colors: mx.array | None = None,
    repair: bool = True,
) -> Path:
    """Author a GLB file at ``out_path``.

    Parameters
    ----------
    vertices : mx.array
        ``[V, 3]`` mesh vertex positions in world space.
    faces : mx.array
        ``[F, 3]`` int32 triangle indices into ``vertices``.
    out_path : str | Path
        Destination path; ``.glb`` extension expected.
    material_colors : mx.array or None
        Optional ``[V, 3]`` per-vertex base color in linear RGB ``[0, 1]``.
        When ``None`` (default) the mesh is exported untextured.
    repair : bool
        When True (default), runs ``trimesh.repair.fix_normals(..., multibody=True)``
        before export to flip back-facing triangles and orient face normals
        outward on each connected component. The Flexible Dual Grid mesh
        extractor does not guarantee consistent winding, so without this
        step many viewers show back-facing triangles as holes due to
        back-face culling. Vertex count, face count, and per-vertex colors
        are preserved — only the column order within each face row may
        change. Set to False to keep the raw extractor output (faster, but
        triangles may appear flipped in renderers that cull back faces).

    Returns
    -------
    Path
        The resolved output path.
    """
    out = Path(out_path)
    verts_np = np.asarray(vertices).astype(np.float32)
    faces_np = np.asarray(faces).astype(np.int32)
    if verts_np.ndim != 2 or verts_np.shape[1] != 3:
        raise ValueError(f"vertices must be [V, 3]; got {verts_np.shape}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"faces must be [F, 3]; got {faces_np.shape}")

    vertex_colors: np.ndarray | None = None
    if material_colors is not None:
        colors_np = np.asarray(material_colors).astype(np.float32)
        if colors_np.shape != verts_np.shape:
            raise ValueError(
                f"material_colors must match vertices shape; "
                f"got {colors_np.shape} vs {verts_np.shape}"
            )
        # trimesh expects uint8 RGBA per vertex.
        vertex_colors = np.concatenate(
            [
                np.clip(colors_np * 255.0, 0, 255).astype(np.uint8),
                np.full((colors_np.shape[0], 1), 255, dtype=np.uint8),
            ],
            axis=1,
        )

    mesh = trimesh.Trimesh(
        vertices=verts_np,
        faces=faces_np,
        vertex_colors=vertex_colors,
        process=False,
    )
    if repair and faces_np.shape[0] > 0:
        # multibody=True so each disjoint piece (e.g. separate gemstones,
        # filigree wires) gets its own outward-orientation pass.
        trimesh.repair.fix_normals(mesh, multibody=True)  # type: ignore[no-untyped-call]
    mesh.export(out)
    return out
