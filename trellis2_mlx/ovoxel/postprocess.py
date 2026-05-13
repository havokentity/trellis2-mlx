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
    fill_holes: bool = False,
    target_faces: int | None = None,
    max_hole_size: int = 30,
    verbose: bool = False,
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
        outward on each connected component. Vertex / face counts and
        per-vertex colors are preserved.
    fill_holes : bool
        When True, runs a ``pymeshlab.meshing_close_holes`` pass to close
        small boundary loops (≤ ``max_hole_size`` edges). Default False.
    target_faces : int or None
        When set, runs quadric edge-collapse decimation down to this face
        count, preserving vertex colors. Use this for a "low-poly" /
        game-ready export. Default None (keep extractor output).
    max_hole_size : int
        Maximum hole perimeter (in edges) to close when ``fill_holes`` is
        True. Default 30.
    verbose : bool
        Print before/after stats during repair / simplification.

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

    vertex_colors_u8: np.ndarray | None = None
    if material_colors is not None:
        colors_np = np.asarray(material_colors).astype(np.float32)
        if colors_np.shape != verts_np.shape:
            raise ValueError(
                f"material_colors must match vertices shape; "
                f"got {colors_np.shape} vs {verts_np.shape}"
            )
        # trimesh expects uint8 RGBA per vertex.
        vertex_colors_u8 = np.concatenate(
            [
                np.clip(colors_np * 255.0, 0, 255).astype(np.uint8),
                np.full((colors_np.shape[0], 1), 255, dtype=np.uint8),
            ],
            axis=1,
        )

    mesh = trimesh.Trimesh(
        vertices=verts_np,
        faces=faces_np,
        vertex_colors=vertex_colors_u8,
        process=False,
    )
    if repair and faces_np.shape[0] > 0:
        # multibody=True so each disjoint piece (e.g. separate gemstones,
        # filigree wires) gets its own outward-orientation pass.
        trimesh.repair.fix_normals(mesh, multibody=True)  # type: ignore[no-untyped-call]

    vertex_normals: np.ndarray | None = None
    if (fill_holes or target_faces is not None) and faces_np.shape[0] > 0:
        # Round-trip through pymeshlab for hole-fill + quadric decimation.
        from trellis2_mlx.ovoxel.repair import repair_mesh

        # Use the (possibly winding-fixed) mesh as input so pymeshlab
        # works on the corrected topology.
        result = repair_mesh(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32),
            vertex_colors=vertex_colors_u8,
            fill_holes=fill_holes,
            max_hole_size=max_hole_size,
            target_faces=target_faces,
            compute_vertex_normals=True,
            verbose=verbose,
        )
        mesh = trimesh.Trimesh(
            vertices=result["vertices"],
            faces=result["faces"],
            vertex_colors=result["vertex_colors"],
            vertex_normals=result["vertex_normals"],
            process=False,
        )
        vertex_normals = result["vertex_normals"]

    if vertex_normals is None and faces_np.shape[0] > 0:
        # Always author per-vertex smooth normals so renderers shade
        # the mesh smoothly. trimesh computes these lazily but doesn't
        # always include them in the GLB unless we touch them.
        _ = mesh.vertex_normals  # trigger computation
    mesh.export(out)
    return out
