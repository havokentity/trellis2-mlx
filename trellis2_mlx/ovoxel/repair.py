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

    def _close_holes() -> None:
        # pymeshlab raises if the mesh has zero boundary loops; swallow
        # that so a clean input passes through unchanged.
        try:
            ms.meshing_close_holes(maxholesize=max_hole_size)
        except Exception as e:  # noqa: BLE001 — pymeshlab raises generic Exception
            if "boundary" not in str(e).lower() and "hole" not in str(e).lower():
                raise

    def _decimate(target: int, aggressive: bool = False) -> None:
        """preservetopology=False is required to actually hit aggressive
        face-count targets on TRELLIS.2 output — the FDG mesh extractor
        produces lots of non-manifold edges (single-voxel-thick filigree,
        interior voids) and with preservetopology=True the decimator
        refuses to collapse across them. We sacrifice topological purity
        for the requested polycount, matching upstream cumesh.simplify.

        When ``aggressive=True``, drops `preservenormal` and lowers the
        quality threshold so the decimator will accept any collapse
        regardless of resulting face quality. Used for retry passes
        when the polite (preservenormal=True, qualitythr=0.3) pass
        refuses to go lower."""
        # qualitythr in pymeshlab is the MINIMUM acceptable quality of
        # the post-collapse triangle — higher = more strict. 0.0 means
        # "accept any collapse" which is what we want when the polite
        # pass got stuck. (Previous code had 1.0 which was max-strict.)
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target),
            preserveboundary=False,
            preservenormal=not aggressive,
            preservetopology=False,
            optimalplacement=True,
            planarquadric=True,
            qualitythr=0.0 if aggressive else 0.3,
        )

    def _cleanup(small_component_size: int = 100, small_component_diameter: float = 0.02) -> None:
        """Standard topology cleanup pass — mirrors upstream cumesh's
        remove_duplicate_faces + repair_non_manifold_edges +
        remove_small_connected_components + fill_holes chain
        (postprocess.py:139-145).

        Drops disconnected islands smaller than ``small_component_size``
        faces (defaults to 100) OR a diameter less than
        ``small_component_diameter`` (in unit-cube coordinates; default
        2% of bbox). Both filters complement each other: the size filter
        kills triangle confetti, the diameter filter kills small but
        face-rich blobs sitting in space.
        """
        import contextlib

        # Each filter may raise if the mesh is in a state it can't act on
        # (e.g. no non-manifold edges, no small components). We treat those
        # as success ("nothing to do") and continue.
        with contextlib.suppress(Exception):
            ms.meshing_remove_duplicate_faces()
        with contextlib.suppress(Exception):
            ms.meshing_remove_duplicate_vertices()
        with contextlib.suppress(Exception):
            ms.meshing_repair_non_manifold_edges()
        with contextlib.suppress(Exception):
            ms.meshing_remove_connected_component_by_face_number(
                mincomponentsize=small_component_size
            )
        with contextlib.suppress(Exception):
            from pymeshlab import PercentageValue
            ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=PercentageValue(small_component_diameter * 100)
            )
        if fill_holes:
            _close_holes()

    # Step 0: weld near-duplicate vertices. The FDG mesh extractor can emit
    # vertices that are spatially coincident but indexed separately (one
    # per voxel rather than one per shared corner). Without welding,
    # every triangle is its own topological component, decimation can't
    # collapse across the implicit seams, and trimesh's
    # `split(only_watertight=False)` sees thousands of "components"
    # that aren't real. 0.05% of bbox diameter is well under one voxel
    # at 512³ source resolution so we don't smear real geometry.
    import contextlib

    with contextlib.suppress(Exception):
        ms.meshing_merge_close_vertices(threshold=ml.PercentageValue(0.05))
    if verbose:
        mm = ms.current_mesh()
        print(f"  after vertex weld: V={mm.vertex_number():,}  F={mm.face_number():,}")

    if fill_holes:
        _close_holes()
        if verbose:
            mm = ms.current_mesh()
            print(f"  after hole-fill: V={mm.vertex_number():,}  F={mm.face_number():,}")

    if target_faces is not None and target_faces > 0:
        # pymeshlab's quadric decimator caps reduction at ~85% per call.
        # For big reductions (cascade 8.4M → 500k = ~94% needed) we
        # iterate quadric, asking for the actual target each pass —
        # the decimator reduces as much as it can per call, and we
        # call it again. When two consecutive passes don't make
        # progress we bail out to clustering decimation which
        # guarantees reduction (spatial vertex binning) at the cost
        # of coarser output.
        #
        # Skipping cleanup BETWEEN passes because cleanup removes
        # non-manifold edges that the decimator needs to collapse
        # aggressively. Without that change the user's cascade test
        # got stuck at 1.4M no matter how many retries we did.
        target = int(target_faces)
        for attempt in range(5):
            current_count = ms.current_mesh().face_number()
            if current_count <= target * 1.2:
                break
            # First pass is polite (preservenormal=True, qualitythr=0.3).
            # Subsequent passes are aggressive (accept any quality, drop
            # the normal-preservation constraint).
            aggressive = attempt > 0
            if verbose:
                print(
                    f"  decimate pass {attempt + 1} "
                    f"({'aggressive' if aggressive else 'polite'}): "
                    f"{current_count:,} → ask for {target:,}"
                )
            prev_count = current_count
            _decimate(target, aggressive=aggressive)
            new_count = ms.current_mesh().face_number()
            if new_count >= prev_count * 0.98:
                # < 2% reduction = quadric is effectively stuck. Fall
                # back to clustering. Threshold heuristic: edge length
                # of a uniform mesh at the target face count, in % of
                # bbox diameter. For unit-cube models with surface area
                # ~6, edge ≈ sqrt(6/N).
                import math

                threshold_pct = math.sqrt(6.0 / target) * 100
                threshold_pct = max(0.05, min(threshold_pct, 5.0))
                if verbose:
                    print(
                        f"    quadric plateau ({prev_count:,} → {new_count:,}); "
                        f"falling back to clustering decimation "
                        f"@ {threshold_pct:.2f}% bbox edge"
                    )
                import contextlib

                with contextlib.suppress(Exception):
                    ms.meshing_decimation_clustering(
                        threshold=ml.PercentageValue(threshold_pct)
                    )
                break

        if verbose:
            mm = ms.current_mesh()
            print(
                f"  after decimate loop:  V={mm.vertex_number():,}  "
                f"F={mm.face_number():,}"
            )

        # Cleanup once, AT THE END. Cleanup between decimation passes
        # was what broke the per-pass reduction: removing dupes /
        # non-manifold / small-components stripped out exactly the
        # edges the decimator wanted to collapse next, leaving a mesh
        # where every face was on a critical edge.
        _cleanup()
        if verbose:
            mm = ms.current_mesh()
            print(
                f"  after cleanup:  V={mm.vertex_number():,}  "
                f"F={mm.face_number():,}"
            )

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
