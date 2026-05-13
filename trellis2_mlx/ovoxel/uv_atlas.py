"""UV-atlas baking: turn per-vertex colors into a proper texture-atlased GLB.

When the material decoder produces per-voxel colors and we attach them as
per-vertex colors in the GLB, every game engine renders them via vertex
interpolation. That works at low poly counts but smears at high poly
counts — and *real* PBR textures (separate base-color + metallic-roughness
images) are what most engines and content pipelines expect.

This module bakes the per-vertex colors into a 2D texture atlas:

1. UV-unwrap the mesh with `xatlas` (a fast LSCM-style chart packer
   that handles non-manifold input — perfect for our FDG output).
2. Software-rasterize each triangle into the atlas image using the
   `xatlas`-provided UV coordinates as 2D positions, with barycentric
   interpolation of vertex colors per pixel.
3. Return (new_verts, new_faces, uvs, texture_image_rgba) plus an
   optional metallic-roughness packed image when those attributes are
   given.

The rasterizer is a per-triangle Python loop using vectorised numpy
inside each triangle's bounding box. For typical decimated meshes
(50k–500k faces, 1024² atlas) this runs in 10-60s — acceptable as a
once-per-export cost. A native Metal kernel could cut this to <1s when
we get to it.
"""

from __future__ import annotations

import numpy as np


def _rasterize_batched(
    uvs_pix: np.ndarray,
    faces: np.ndarray,
    attrs: np.ndarray,
    img: np.ndarray,
    mask: np.ndarray,
    chunk_size: int = 8192,
) -> None:
    """Vectorized rasterizer — processes all triangles via numpy fancy indexing.

    Replaces the per-triangle Python loop. For every triangle we generate
    its bounding-box pixels into a single flat array, run barycentric
    interpolation in one batch, mask inside-triangle pixels, and scatter
    into ``img``.

    Memory: each chunk holds up to ``chunk_size * (max_bbox_pixels)``
    intermediate pixels. With chart-packed UV atlases, triangles are
    small (tens of pixels), so the per-chunk working set is bounded.

    Parameters
    ----------
    uvs_pix : ``[V, 2]`` float32 pixel-space UV positions.
    faces : ``[F, 3]`` int32 vertex indices.
    attrs : ``[V, C]`` per-vertex attributes (uint8 or float).
    img : ``[H, W, C]`` output image, written in-place.
    mask : ``[H, W]`` bool coverage mask, written in-place.
    chunk_size : Max number of triangles to process at once. Lower this
                 if memory is a concern at very high atlas resolutions.
    """
    img_h, img_w, img_c = img.shape
    n_faces = faces.shape[0]
    if n_faces == 0:
        return

    # Pre-compute everything we need per-triangle.
    tri_uvs = uvs_pix[faces]  # [F, 3, 2]
    tri_attrs = attrs[faces].astype(np.float32)  # [F, 3, C]

    # Bbox per triangle, clipped to image.
    x_min_all = np.clip(np.floor(tri_uvs[:, :, 0].min(axis=1)).astype(np.int32), 0, img_w - 1)
    x_max_all = np.clip(np.ceil(tri_uvs[:, :, 0].max(axis=1)).astype(np.int32), 0, img_w - 1)
    y_min_all = np.clip(np.floor(tri_uvs[:, :, 1].min(axis=1)).astype(np.int32), 0, img_h - 1)
    y_max_all = np.clip(np.ceil(tri_uvs[:, :, 1].max(axis=1)).astype(np.int32), 0, img_h - 1)
    bbox_w = (x_max_all - x_min_all + 1).clip(min=0)
    bbox_h = (y_max_all - y_min_all + 1).clip(min=0)

    # Signed area for each triangle.
    v0 = tri_uvs[:, 0]
    v1 = tri_uvs[:, 1]
    v2 = tri_uvs[:, 2]
    area_all = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (
        v2[:, 0] - v0[:, 0]
    )
    nondegen = np.abs(area_all) > 1e-10

    # Process triangles in chunks to bound memory. Each chunk concatenates
    # all the per-triangle pixel coordinates into a flat array and
    # computes barycentric in one big numpy call.
    eps = 1e-6
    for start in range(0, n_faces, chunk_size):
        end = min(start + chunk_size, n_faces)
        # Filter to non-degenerate triangles with non-empty bbox in this chunk.
        chunk_idx = np.arange(start, end)
        valid = nondegen[start:end] & (bbox_w[start:end] > 0) & (bbox_h[start:end] > 0)
        chunk_idx = chunk_idx[valid]
        if chunk_idx.size == 0:
            continue

        x_min = x_min_all[chunk_idx]
        x_max = x_max_all[chunk_idx]
        y_min = y_min_all[chunk_idx]
        y_max = y_max_all[chunk_idx]
        bw = (x_max - x_min + 1).astype(np.int64)
        bh = (y_max - y_min + 1).astype(np.int64)
        n_pix_per_tri = bw * bh  # [N_chunk]
        total_pix = int(n_pix_per_tri.sum())
        if total_pix == 0:
            continue

        # Build flat (px, py, tri_local_idx) arrays via repeat.
        tri_local_idx = np.repeat(np.arange(chunk_idx.size, dtype=np.int64), n_pix_per_tri)
        # Per-pixel offset within each triangle's bbox: [0, n_pix_per_tri[i]).
        offsets = np.arange(total_pix, dtype=np.int64) - np.repeat(
            np.concatenate([[0], np.cumsum(n_pix_per_tri[:-1])]), n_pix_per_tri
        )
        bw_per_pix = bw[tri_local_idx]
        local_dx = offsets % bw_per_pix
        local_dy = offsets // bw_per_pix
        px = x_min[tri_local_idx] + local_dx
        py = y_min[tri_local_idx] + local_dy

        # Barycentric interp. Pixel centers at +0.5.
        pf = px.astype(np.float32) + 0.5
        pyf = py.astype(np.float32) + 0.5
        tri_real_idx = chunk_idx[tri_local_idx]
        v0x = v0[tri_real_idx, 0]
        v0y = v0[tri_real_idx, 1]
        v1x = v1[tri_real_idx, 0]
        v1y = v1[tri_real_idx, 1]
        v2x = v2[tri_real_idx, 0]
        v2y = v2[tri_real_idx, 1]
        inv_area = 1.0 / area_all[tri_real_idx]
        # w0 = area(P, v1, v2) / total_area
        w0 = ((v1x - pf) * (v2y - pyf) - (v1y - pyf) * (v2x - pf)) * inv_area
        w1 = ((v2x - pf) * (v0y - pyf) - (v2y - pyf) * (v0x - pf)) * inv_area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        if not inside.any():
            continue

        # Apply mask. Keep only inside pixels that aren't already covered.
        not_yet_covered = ~mask[py[inside], px[inside]]
        inside_idx = np.where(inside)[0]
        write_idx = inside_idx[not_yet_covered]
        if write_idx.size == 0:
            continue
        wpx = px[write_idx]
        wpy = py[write_idx]
        # Barycentric-interpolate attributes for these pixels.
        a0 = tri_attrs[tri_real_idx[write_idx], 0]
        a1 = tri_attrs[tri_real_idx[write_idx], 1]
        a2 = tri_attrs[tri_real_idx[write_idx], 2]
        col = (
            w0[write_idx, None] * a0 + w1[write_idx, None] * a1 + w2[write_idx, None] * a2
        )
        img[wpy, wpx] = np.clip(col, 0, 255).astype(img.dtype)
        mask[wpy, wpx] = True
    _ = img_c  # silence unused-var; img is mutated in place


def bake_uv_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    texture_size: int = 1024,
    *,
    metallic: np.ndarray | None = None,
    roughness: np.ndarray | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """UV-unwrap + per-pixel bake.

    Parameters
    ----------
    vertices : ``[V, 3]`` float
    faces : ``[F, 3]`` int
    vertex_colors : ``[V, 3]`` or ``[V, 4]`` uint8 RGB(A) base-color per vertex.
    texture_size : int
        Atlas resolution. 1024 / 2048 / 4096 are typical. Larger = sharper
        but slower bake + bigger GLB.
    metallic : ``[V, 1]`` or ``[V]`` uint8, optional. Packed into B channel
        of the metallic-roughness texture (glTF convention).
    roughness : ``[V, 1]`` or ``[V]`` uint8, optional. Packed into G channel.

    Returns
    -------
    dict with keys:
        ``vertices`` ``[V', 3]`` — atlas-duplicated verts.
        ``faces`` ``[F, 3]`` — face indices into ``vertices``.
        ``uvs`` ``[V', 2]`` — UV coords in ``[0, 1]``.
        ``base_color`` ``[H, W, 4]`` uint8 RGBA texture image.
        ``metallic_roughness`` ``[H, W, 3]`` uint8 BGR texture image (None if
            both ``metallic`` and ``roughness`` are None).
    """
    import xatlas

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be [V, 3]; got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be [F, 3]; got {faces.shape}")
    if vertex_colors.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"vertex_colors[0]={vertex_colors.shape[0]} but vertices[0]={vertices.shape[0]}"
        )
    if vertex_colors.shape[1] not in (3, 4):
        raise ValueError(f"vertex_colors must be [V, 3] or [V, 4]; got {vertex_colors.shape}")

    n_in_verts = vertices.shape[0]
    n_in_faces = faces.shape[0]
    if verbose:
        print(f"  UV unwrap input: V={n_in_verts:,}  F={n_in_faces:,}")

    # xatlas.parametrize requires C-contiguous float32 verts + uint32 faces.
    verts_in = np.ascontiguousarray(vertices, dtype=np.float32)
    faces_in = np.ascontiguousarray(faces, dtype=np.uint32)
    vmapping, faces_atlas, uvs = xatlas.parametrize(verts_in, faces_in)
    # vmapping: [V', 1] uint32 — atlas vertex idx → original vertex idx
    # faces_atlas: [F, 3] uint32
    # uvs: [V', 2] float32 in [0, 1]
    n_atlas_verts = vmapping.shape[0]
    if verbose:
        print(
            f"  UV unwrap done:  V'={n_atlas_verts:,} ({n_atlas_verts/n_in_verts:.2f}× dup), "
            f"F={faces_atlas.shape[0]:,}"
        )

    # Resolve duplicated verts → original attributes via vmapping.
    new_verts = verts_in[vmapping]
    # Promote 3-channel colors to RGBA.
    if vertex_colors.shape[1] == 3:
        colors_rgba = np.concatenate(
            [vertex_colors, np.full((n_in_verts, 1), 255, dtype=np.uint8)], axis=1
        )
    else:
        colors_rgba = vertex_colors.astype(np.uint8)
    new_colors = colors_rgba[vmapping]
    new_metallic = metallic[vmapping] if metallic is not None else None
    new_roughness = roughness[vmapping] if roughness is not None else None

    # Rasterize colors into the atlas image.
    tex_h = tex_w = int(texture_size)
    base_color_img = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
    base_color_img[:, :, 3] = 255  # opaque background (overwritten by mask)
    coverage = np.zeros((tex_h, tex_w), dtype=bool)

    # Convert UVs from [0, 1] to pixel space. glTF UV origin is top-left
    # with v growing down — xatlas hands us v growing up, so flip.
    uvs_pix = np.empty_like(uvs)
    uvs_pix[:, 0] = uvs[:, 0] * (tex_w - 1)
    uvs_pix[:, 1] = (1.0 - uvs[:, 1]) * (tex_h - 1)

    has_mr = metallic is not None or roughness is not None
    mr_img: np.ndarray | None = None
    if has_mr:
        mr_img = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)

    # Batched, vectorized rasterization across ALL triangles. Each chunk
    # holds up to ``chunk_size`` triangles' bbox pixels and runs the
    # barycentric test + interp in one big numpy call. This replaces a
    # per-triangle Python loop that was ~500 µs / triangle and ran into
    # multi-minute bakes at >100k faces.
    n_faces = faces_atlas.shape[0]
    if verbose:
        print(f"    rasterising {n_faces:,} triangles (vectorised)...")
    _rasterize_batched(uvs_pix, faces_atlas, new_colors, base_color_img, coverage)

    if has_mr and mr_img is not None:
        # glTF metallic-roughness channel layout: R unused, G=roughness, B=metallic.
        # Build a per-vertex MR attribute array, then rasterize the same way.
        nv_atlas = new_colors.shape[0]
        mr_per_vert = np.zeros((nv_atlas, 3), dtype=np.uint8)
        if new_roughness is not None:
            mr_per_vert[:, 1] = new_roughness.astype(np.uint8).flatten()
        else:
            mr_per_vert[:, 1] = 128
        if new_metallic is not None:
            mr_per_vert[:, 2] = new_metallic.astype(np.uint8).flatten()
        mr_coverage = np.zeros((tex_h, tex_w), dtype=bool)
        _rasterize_batched(uvs_pix, faces_atlas, mr_per_vert, mr_img, mr_coverage)

    # Pad the atlas: any pixel inside a chart that's still uncovered gets the
    # nearest covered pixel's value. This is the standard texture-bleed
    # mitigation — without it, bilinear filtering at chart edges pulls in
    # zeroed pixels and creates dark seams. Implement as a simple iterative
    # dilation (numpy-only).
    base_color_img = _dilate_coverage(base_color_img, coverage, iterations=8)
    if mr_img is not None:
        mr_img = _dilate_coverage(mr_img, coverage, iterations=8)

    if verbose:
        cov_frac = coverage.mean()
        print(f"  rasterisation done: coverage={cov_frac*100:.1f}%")

    return {
        "vertices": new_verts,
        "faces": faces_atlas.astype(np.int32),
        "uvs": uvs.astype(np.float32),
        "base_color": base_color_img,
        "metallic_roughness": mr_img if mr_img is not None else np.zeros((1, 1, 3), dtype=np.uint8),
    }


def _dilate_coverage(img: np.ndarray, mask: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Bleed covered pixels into adjacent uncovered pixels — fixes UV seams.

    Each iteration: every uncovered pixel that has a 4-connected neighbor
    in the covered set gets that neighbor's value (priority: left, right,
    up, down). Marks the newly-filled pixel as covered for subsequent
    iterations.
    """
    img = img.copy()
    mask = mask.copy()
    for _ in range(iterations):
        # Build the previous-iteration mask before updates so we expand
        # one ring at a time.
        prev = mask.copy()
        # 4 directional shifts: take values from each direction where source
        # was covered last iteration.
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            shifted_img = np.roll(img, shift, axis=axis)
            shifted_mask = np.roll(prev, shift, axis=axis)
            # Don't bleed across the wrap edge.
            if axis == 0:
                if shift > 0:
                    shifted_mask[:shift] = False
                else:
                    shifted_mask[shift:] = False
            else:
                if shift > 0:
                    shifted_mask[:, :shift] = False
                else:
                    shifted_mask[:, shift:] = False

            fill_here = shifted_mask & ~mask  # uncovered pixels with covered neighbor
            img[fill_here] = shifted_img[fill_here]
            mask |= fill_here
    return img


__all__ = ["bake_uv_atlas"]
