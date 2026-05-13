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


def _rasterize_triangle(
    img: np.ndarray,
    mask: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
    attr0: np.ndarray,
    attr1: np.ndarray,
    attr2: np.ndarray,
) -> None:
    """In-place: rasterize one triangle into ``img`` with barycentric attr interp.

    ``img`` shape: ``[H, W, C]``.
    ``uv0/uv1/uv2``: ``[2]`` pixel-space coords (origin top-left, y down).
    ``attr0/attr1/attr2``: ``[C]`` per-vertex attributes; output is barycentric
    interpolated.
    """
    img_h, img_w, _img_c = img.shape
    # Bounding box (inclusive).
    x_min = max(0, int(np.floor(min(uv0[0], uv1[0], uv2[0]))))
    x_max = min(img_w - 1, int(np.ceil(max(uv0[0], uv1[0], uv2[0]))))
    y_min = max(0, int(np.floor(min(uv0[1], uv1[1], uv2[1]))))
    y_max = min(img_h - 1, int(np.ceil(max(uv0[1], uv1[1], uv2[1]))))
    if x_max < x_min or y_max < y_min:
        return

    # Pixel centers in the bbox.
    xs = np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5
    ys = np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5
    px, py = np.meshgrid(xs, ys)  # both [Hb, Wb]

    # Barycentric coords using the signed-area / sub-triangles formula.
    # alpha = area(P, v1, v2) / area(v0, v1, v2)
    # beta  = area(v0, P, v2) / area(...)
    # gamma = 1 - alpha - beta
    def edge(ax: float, ay: float, bx: float, by: float, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    area = edge(uv0[0], uv0[1], uv1[0], uv1[1], np.array(uv2[0]), np.array(uv2[1]))
    if abs(float(area)) < 1e-10:
        return  # degenerate
    inv_area = 1.0 / float(area)

    w0 = edge(uv1[0], uv1[1], uv2[0], uv2[1], px, py) * inv_area
    w1 = edge(uv2[0], uv2[1], uv0[0], uv0[1], px, py) * inv_area
    w2 = 1.0 - w0 - w1

    # Conservative inside test — include edges (with small epsilon for AA-ish).
    eps = 1e-6
    inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)

    if not inside.any():
        return

    # Interpolate attributes — [Hb, Wb, C].
    attr = (
        w0[..., None] * attr0[None, None, :]
        + w1[..., None] * attr1[None, None, :]
        + w2[..., None] * attr2[None, None, :]
    )

    # Write to img. Indexing the slice region is OK — we mask via `inside`.
    region = img[y_min : y_max + 1, x_min : x_max + 1]
    mask_region = mask[y_min : y_max + 1, x_min : x_max + 1]
    # Only write where not already covered, to keep first-write deterministic.
    write_mask = inside & ~mask_region
    region[write_mask] = attr[write_mask].astype(region.dtype)
    mask_region |= inside


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
    mr_coverage: np.ndarray | None = None
    if has_mr:
        mr_img = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
        mr_coverage = np.zeros((tex_h, tex_w), dtype=bool)

    n_faces = faces_atlas.shape[0]
    log_every = max(1, n_faces // 10)
    for f_idx in range(n_faces):
        i0, i1, i2 = faces_atlas[f_idx]
        _rasterize_triangle(
            base_color_img,
            coverage,
            uvs_pix[i0],
            uvs_pix[i1],
            uvs_pix[i2],
            new_colors[i0],
            new_colors[i1],
            new_colors[i2],
        )
        if has_mr and mr_img is not None:
            # glTF MR channel layout: R unused, G=roughness, B=metallic.
            mr0 = np.array(
                [
                    0,
                    int(new_roughness[i0].item() if new_roughness is not None else 128),
                    int(new_metallic[i0].item() if new_metallic is not None else 0),
                ],
                dtype=np.uint8,
            )
            mr1 = np.array(
                [
                    0,
                    int(new_roughness[i1].item() if new_roughness is not None else 128),
                    int(new_metallic[i1].item() if new_metallic is not None else 0),
                ],
                dtype=np.uint8,
            )
            mr2 = np.array(
                [
                    0,
                    int(new_roughness[i2].item() if new_roughness is not None else 128),
                    int(new_metallic[i2].item() if new_metallic is not None else 0),
                ],
                dtype=np.uint8,
            )
            assert mr_coverage is not None
            _rasterize_triangle(
                mr_img, mr_coverage,
                uvs_pix[i0], uvs_pix[i1], uvs_pix[i2],
                mr0, mr1, mr2,
            )
        if verbose and (f_idx + 1) % log_every == 0:
            cov_frac = coverage.mean()
            print(
                f"    rasterised {f_idx + 1:,}/{n_faces:,} faces  coverage={cov_frac*100:.1f}%"
            )

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
