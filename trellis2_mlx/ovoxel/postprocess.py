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


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals. Pure numpy; avoids trimesh's
    scipy-backed implementation so we don't need scipy as a runtime dep.

    Returns ``[V, 3] float32`` unit vectors. Vertices not referenced by
    any face come out as zeros (those happen e.g. after some decimation
    paths leave detached vertices)."""
    face_normals = np.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]],
        verts[faces[:, 2]] - verts[faces[:, 0]],
    )
    vertex_normals = np.zeros_like(verts)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    return (vertex_normals / np.maximum(norms, 1e-8)).astype(np.float32)


def export_glb(
    vertices: mx.array,
    faces: mx.array,
    out_path: str | Path,
    *,
    material_colors: mx.array | None = None,
    metallic: mx.array | None = None,
    roughness: mx.array | None = None,
    repair: bool = True,
    fill_holes: bool = False,
    target_faces: int | None = None,
    max_hole_size: int = 30,
    uv_atlas: bool = False,
    texture_size: int = 1024,
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

    # Order matters: decimate FIRST so the (expensive, single-threaded numpy
    # flood-fill in trimesh.repair.fix_normals) operates on the small mesh.
    # On the cascade output (8.3M faces → 500k target), this is the difference
    # between ~5 min and ~30 s for the repair stage. We don't lose anything
    # because pymeshlab's quadric decimator with preservenormal=True keeps
    # whatever winding direction it found in the input mesh, and fix_normals
    # downstream propagates that to its proper outward orientation across
    # adjacent face components.
    vertex_normals: np.ndarray | None = None
    if (fill_holes or target_faces is not None) and faces_np.shape[0] > 0:
        from trellis2_mlx.ovoxel.repair import repair_mesh

        result = repair_mesh(
            verts_np,
            faces_np,
            vertex_colors=vertex_colors_u8,
            fill_holes=fill_holes,
            max_hole_size=max_hole_size,
            target_faces=target_faces,
            compute_vertex_normals=True,
            verbose=verbose,
        )
        verts_np = result["vertices"]
        faces_np = result["faces"]
        vertex_colors_u8 = result["vertex_colors"]
        vertex_normals = result["vertex_normals"]

    mesh = trimesh.Trimesh(
        vertices=verts_np,
        faces=faces_np,
        vertex_colors=vertex_colors_u8,
        vertex_normals=vertex_normals,
        process=False,
    )

    if repair and faces_np.shape[0] > 0:
        # multibody=True so each disjoint piece (e.g. separate gemstones,
        # filigree wires) gets its own outward-orientation pass. After
        # decimation the mesh has far fewer faces so this is fast.
        trimesh.repair.fix_normals(mesh, multibody=True)  # type: ignore[no-untyped-call]
        # fix_normals only changes face winding (column-order within each
        # face row), not vertex positions or count, so pymeshlab-computed
        # vertex_normals stay valid — but their direction may be inverted
        # for faces whose winding flipped. Re-derive on the fly.
        if vertex_normals is not None:
            _ = mesh.vertex_normals  # force trimesh recompute from new faces

    if vertex_normals is None and faces_np.shape[0] > 0:
        # Always author per-vertex smooth normals so renderers shade
        # the mesh smoothly. We compute area-weighted normals in pure
        # numpy because trimesh's path goes through scipy's sparse
        # matrix and we don't want a scipy runtime dep just for this.
        current_v = np.asarray(mesh.vertices, dtype=np.float32)
        current_f = np.asarray(mesh.faces, dtype=np.int32)
        normals = _compute_vertex_normals(current_v, current_f)
        # Rebuild the mesh with explicit normals so trimesh's export
        # writes the buffer view without trying to recompute on its own.
        existing_vc: np.ndarray | None = None
        vis = mesh.visual
        if vis is not None:
            vc = getattr(vis, "vertex_colors", None)
            if vc is not None:
                existing_vc = np.asarray(vc)
        mesh = trimesh.Trimesh(
            vertices=current_v,
            faces=current_f,
            vertex_colors=existing_vc,
            vertex_normals=normals,
            process=False,
        )

    if uv_atlas and faces_np.shape[0] > 0 and vertex_colors_u8 is not None:
        # UV-unwrap + bake per-vertex colors into a 2D texture atlas.
        # Replaces the per-vertex color attribute with a proper PBR material
        # so engines that use texture sampling (i.e. all of them) get
        # higher-quality, decimation-friendly textures.
        from trellis2_mlx.ovoxel.uv_atlas import bake_uv_atlas

        metallic_np: np.ndarray | None = None
        roughness_np: np.ndarray | None = None
        if metallic is not None:
            m = np.asarray(metallic).astype(np.float32).flatten()
            metallic_np = np.clip(m * 255.0, 0, 255).astype(np.uint8)
        if roughness is not None:
            r = np.asarray(roughness).astype(np.float32).flatten()
            roughness_np = np.clip(r * 255.0, 0, 255).astype(np.uint8)

        # If a decimation pass ran, vertex_colors_u8 was updated; otherwise
        # we still hold the input colors. mesh.vertices/faces hold the
        # current geometry after any repair pass.
        current_verts = np.asarray(mesh.vertices, dtype=np.float32)
        current_faces = np.asarray(mesh.faces, dtype=np.int32)
        # If repair flipped winding we want to use the flipped colors; map
        # by the surviving vertex array (vertex count unchanged by repair).
        atlas = bake_uv_atlas(
            current_verts,
            current_faces,
            vertex_colors_u8 if vertex_colors_u8.shape[0] == current_verts.shape[0]
            else vertex_colors_u8[: current_verts.shape[0]],
            texture_size=texture_size,
            metallic=metallic_np[: current_verts.shape[0]] if metallic_np is not None else None,
            roughness=roughness_np[: current_verts.shape[0]] if roughness_np is not None else None,
            verbose=verbose,
        )
        # Author the texture atlas via trimesh's TextureVisuals.
        from PIL import Image as _PILImage

        base_color_pil = _PILImage.fromarray(atlas["base_color"], mode="RGBA")
        mr_arr = atlas.get("metallic_roughness")
        material = trimesh.visual.material.PBRMaterial(  # type: ignore[no-untyped-call]
            baseColorTexture=base_color_pil,
            metallicFactor=0.0 if metallic is None else 1.0,
            roughnessFactor=1.0 if roughness is None else 1.0,
        )
        if mr_arr is not None and mr_arr.size > 9:
            mr_pil = _PILImage.fromarray(mr_arr, mode="RGB")
            material.metallicRoughnessTexture = mr_pil
        mesh = trimesh.Trimesh(
            vertices=atlas["vertices"],
            faces=atlas["faces"],
            visual=trimesh.visual.TextureVisuals(  # type: ignore[no-untyped-call]
                uv=atlas["uvs"], material=material
            ),
            process=False,
        )

    mesh.export(out)
    return out
