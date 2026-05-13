"""Mesh post-processing chain: hole-fill + quadric decimation + smooth normals.

This is our Apple-Silicon equivalent of upstream's CUDA `cumesh` chain
(see ``reference/microsoft-trellis2/o-voxel/o_voxel/postprocess.py``).
Upstream uses ``cumesh.CuMesh.fill_holes`` + ``cumesh.CuMesh.simplify`` +
``cumesh.CuMesh.repair_non_manifold_edges``; we use ``pymeshlab`` (CGAL
bindings, CPU) to achieve the same result without CUDA. The output is
suitable for game engines: clean topology, target poly count, smooth
per-vertex normals.

The chain (when all knobs are on):

1. ``meshing_close_holes`` — close boundary loops up to ``max_hole_size``
   edges. Larger holes are left for explicit handling.
2. ``meshing_decimation_quadric_edge_collapse`` — Garland-Heckbert
   quadric error metric edge-collapse decimation, preserving vertex
   colors and surface normals.
3. ``compute_normal_per_vertex`` — re-compute smoothed per-vertex
   normals on the simplified mesh so renderers shade it smoothly even
   at low poly counts.

Per-vertex colors are passed through end-to-end: ``pymeshlab`` carries
them as a ``v_color_matrix`` and interpolates correctly during the
quadric collapse, so the result has one color per surviving vertex
with no further work.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class RepairResult(TypedDict):
    """Output of :func:`repair_mesh`. Vertex / face counts can change."""

    vertices: np.ndarray  # [V', 3] float32
    faces: np.ndarray  # [F', 3] int32
    vertex_colors: np.ndarray | None  # [V', 4] uint8 RGBA or None
    vertex_normals: np.ndarray | None  # [V', 3] float32 or None


def repair_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None = None,
    *,
    fill_holes: bool = True,
    max_hole_size: int = 30,
    target_faces: int | None = None,
    compute_vertex_normals: bool = True,
    verbose: bool = False,
) -> RepairResult:
    """Run the post-processing chain on a triangle mesh.

    Parameters
    ----------
    vertices : np.ndarray
        ``[V, 3]`` vertex positions, any float dtype.
    faces : np.ndarray
        ``[F, 3]`` triangle indices.
    vertex_colors : np.ndarray or None
        Optional ``[V, 4]`` uint8 RGBA per-vertex colors. Preserved
        (interpolated) through decimation.
    fill_holes : bool
        Close boundary loops up to ``max_hole_size`` edges. Default True.
    max_hole_size : int
        Maximum hole perimeter (in edges) to close. Larger boundaries
        are left as-is. Upstream's ``cumesh.fill_holes`` uses a
        perimeter-in-unit-cube threshold of 3e-2; this is rougher but
        works in practice.
    target_faces : int or None
        If given, run quadric edge-collapse decimation down to this
        face count. ``None`` skips the simplification step.
    compute_vertex_normals : bool
        Re-compute smooth per-vertex normals on the final mesh so the
        GLB renders with smooth shading. Default True.
    verbose : bool
        Print before/after stats. Default False.

    Returns
    -------
    RepairResult
        New geometry and (optionally) colors / normals.
    """
    import pymeshlab as ml

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be [V, 3]; got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be [F, 3]; got {faces.shape}")
    if vertex_colors is not None:
        if vertex_colors.shape[0] != vertices.shape[0]:
            raise ValueError(
                f"vertex_colors[0]={vertex_colors.shape[0]} but vertices[0]={vertices.shape[0]}"
            )
        if vertex_colors.shape[1] not in (3, 4):
            raise ValueError(f"vertex_colors must be [V, 3] or [V, 4]; got {vertex_colors.shape}")

    if verbose:
        print(
            f"  repair input:  V={vertices.shape[0]:,}  F={faces.shape[0]:,}"
            f"  colors={'yes' if vertex_colors is not None else 'no'}"
        )

    ms = ml.MeshSet()
    v64 = np.ascontiguousarray(vertices, dtype=np.float64)
    f32 = np.ascontiguousarray(faces, dtype=np.int32)
    if vertex_colors is not None:
        # uint8 RGBA → float64 RGBA in [0, 1]. Promote RGB to RGBA if
        # the caller handed us 3-channel colors.
        if vertex_colors.shape[1] == 3:
            rgba_u8 = np.concatenate(
                [vertex_colors, np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)],
                axis=1,
            )
        else:
            rgba_u8 = vertex_colors
        c64 = np.ascontiguousarray(rgba_u8, dtype=np.float64) / 255.0
        m = ml.Mesh(vertex_matrix=v64, face_matrix=f32, v_color_matrix=c64)
    else:
        m = ml.Mesh(vertex_matrix=v64, face_matrix=f32)
    ms.add_mesh(m)

    if fill_holes:
        # pymeshlab raises if the mesh has zero boundary loops; swallow
        # that so a clean input passes through unchanged.
        try:
            ms.meshing_close_holes(maxholesize=max_hole_size)
        except Exception as e:  # noqa: BLE001 — pymeshlab raises generic Exception
            if "boundary" not in str(e).lower() and "hole" not in str(e).lower():
                raise
        if verbose:
            mm = ms.current_mesh()
            print(f"  after hole-fill: V={mm.vertex_number():,}  F={mm.face_number():,}")

    if target_faces is not None and target_faces > 0:
        # preservetopology=False is required to actually hit aggressive
        # face-count targets on TRELLIS.2 output — the FDG mesh extractor
        # produces lots of non-manifold edges (single-voxel-thick filigree,
        # interior voids) and with preservetopology=True the decimator
        # refuses to collapse across them. We sacrifice topological purity
        # for the requested polycount, matching the upstream cumesh.simplify
        # behavior.
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            preserveboundary=False,
            preservenormal=True,
            preservetopology=False,
            optimalplacement=True,
            planarquadric=True,
            qualitythr=0.3,
        )
        if verbose:
            mm = ms.current_mesh()
            print(f"  after simplify:  V={mm.vertex_number():,}  F={mm.face_number():,}")

    if compute_vertex_normals:
        ms.compute_normal_per_vertex()

    final = ms.current_mesh()
    new_v = np.ascontiguousarray(final.vertex_matrix(), dtype=np.float32)
    new_f = np.ascontiguousarray(final.face_matrix(), dtype=np.int32)

    new_colors: np.ndarray | None = None
    if vertex_colors is not None:
        c01 = final.vertex_color_matrix()
        new_colors = (np.clip(c01, 0.0, 1.0) * 255.0).astype(np.uint8)

    new_normals: np.ndarray | None = None
    if compute_vertex_normals:
        new_normals = np.ascontiguousarray(final.vertex_normal_matrix(), dtype=np.float32)

    return RepairResult(
        vertices=new_v,
        faces=new_f,
        vertex_colors=new_colors,
        vertex_normals=new_normals,
    )


__all__ = ["RepairResult", "repair_mesh"]
