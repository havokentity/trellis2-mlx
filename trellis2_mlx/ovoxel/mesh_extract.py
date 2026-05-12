"""Flexible Dual Grid → triangle mesh extraction.

Implements the inference path of
``reference/microsoft-trellis2/o-voxel/o_voxel/convert/flexible_dual_grid.py:flexible_dual_grid_to_mesh``.

Algorithm:

1. Each active voxel emits one **dual vertex** at world coord
   ``(coord + v) * voxel_size + aabb_min``.
2. For each ``(voxel, axis)`` pair where ``δ_logits[voxel, axis] > 0``,
   look up the 4 neighbor voxels around that axis-edge. If all 4 are in
   the active set, emit a **quad** of dual vertices.
3. Each quad is split into 2 triangles. The diagonal is chosen by
   comparing ``γ[i_0] * γ[i_2]`` vs ``γ[i_1] * γ[i_3]`` — see
   ``docs/open-questions-resolved.md`` Q7.

This is the **inference** path. Training uses a 4-triangle-per-quad
split with a soft midpoint vertex; the diff-rast loss runs through
``trellis2/representations/mesh`` and ``mtldiffrast``, neither of
which is needed for inference. Our current implementation is
numpy-backed (one big batched lookup per axis) — the Metal-kernel
replacement (``mesh_extract.metal``) lands with Phase 1 step 6 in
``PHASE0_SPEC.md §9``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

# Per-axis ring offsets — match upstream `edge_neighbor_voxel_offset` exactly.
# Axis k corresponds to slot k of the decoder's δ output.
_EDGE_NEIGHBOR_OFFSETS = np.array(
    [
        # 4 voxel offsets around the axis-0 edge (z held constant)
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        # axis-1 edge (y held constant)
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        # axis-2 edge (x held constant)
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    dtype=np.int64,
)  # [3, 4, 3]

# Two candidate diagonal splits — match upstream `quad_split_1` / `quad_split_2`.
_QUAD_SPLIT_DIAG_02 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)  # diagonal 0-2
_QUAD_SPLIT_DIAG_13 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int64)  # diagonal 1-3


def extract_mesh(
    coords: mx.array,
    v: mx.array,
    delta_logits: mx.array,
    gamma: mx.array,
    *,
    grid_size: int,
    aabb_min: tuple[float, float, float] = (-0.5, -0.5, -0.5),
    aabb_max: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[mx.array, mx.array]:
    """Extract a triangle mesh from a Flexible Dual Grid O-Voxel.

    Parameters
    ----------
    coords : mx.array
        ``[L, 3]`` int voxel coordinates at the output resolution.
    v : mx.array
        ``[L, 3]`` dual-vertex offsets (in
        ``[-voxel_margin, 1 + voxel_margin]``).
    delta_logits : mx.array
        ``[L, 3]`` raw edge-activity logits. Edges with logit > 0 are
        kept; ``δ[i, axis] = True`` means the axis-``axis`` edge of voxel
        ``i`` is "intersected" and the dual quad around it should be
        emitted as 2 triangles.
    gamma : mx.array
        ``[L, 1]`` per-voxel quad-split weight. The diagonal of each
        quad is chosen by comparing ``γ[i_0]·γ[i_2]`` vs ``γ[i_1]·γ[i_3]``.
    grid_size : int
        Output grid resolution ``N``. Used to bounds-check neighbor coords
        and to encode the spatial hashmap.
    aabb_min, aabb_max : tuple of 3 floats
        World-space bounds. ``(coord + v) * voxel_size + aabb_min`` maps
        voxel coords into world space. Default ``[-0.5, 0.5]`` is the
        upstream default for the inference pipeline.

    Returns
    -------
    vertices : mx.array
        ``[L, 3]`` float vertex positions in world space (one dual vertex
        per active voxel). Note: not all of these are referenced by faces —
        a voxel with no active δ flag contributes a vertex but no faces.
    faces : mx.array
        ``[M, 3]`` int32 triangle indices into ``vertices``. Each quad
        contributes 2 triangles, so ``M = 2 * (number of valid quads)``.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [L, 3]; got {tuple(coords.shape)}")
    n_active = coords.shape[0]
    if not (
        v.shape == (n_active, 3)
        and delta_logits.shape == (n_active, 3)
        and gamma.shape == (n_active, 1)
    ):
        raise ValueError(
            f"shape mismatch: coords={tuple(coords.shape)} v={tuple(v.shape)} "
            f"delta={tuple(delta_logits.shape)} gamma={tuple(gamma.shape)}"
        )

    coords_np = np.asarray(coords).astype(np.int64)
    v_np = np.asarray(v).astype(np.float32)
    delta_np = np.asarray(delta_logits).astype(np.float32) > 0  # [L, 3] bool
    gamma_np = np.asarray(gamma).astype(np.float32).reshape(-1)  # [L]

    aabb_min_arr = np.asarray(aabb_min, dtype=np.float32).reshape(1, 3)
    aabb_max_arr = np.asarray(aabb_max, dtype=np.float32).reshape(1, 3)
    voxel_size_arr = (aabb_max_arr - aabb_min_arr) / float(grid_size)  # [1, 3]

    # 1. Dual vertices in world space.
    mesh_vertices = (coords_np.astype(np.float32) + v_np) * voxel_size_arr + aabb_min_arr

    # No active edges → no faces.
    if not delta_np.any():
        return mx.array(mesh_vertices), mx.zeros((0, 3), dtype=mx.int32)

    # 2. For each (voxel, axis) with δ True, gather the 4 ring-voxel coords.
    # edge_neighbor_voxel: [L, 3, 4, 3]
    edge_neighbor_voxel = coords_np[:, None, None, :] + _EDGE_NEIGHBOR_OFFSETS[None, :, :, :]
    connected_voxel = edge_neighbor_voxel[delta_np]  # [M, 4, 3]
    m_edges = connected_voxel.shape[0]

    # 3. Spatial hash lookup (same searchsorted approach as build_neighbor_table).
    keys_active = (
        coords_np[:, 0] * (grid_size * grid_size) + coords_np[:, 1] * grid_size + coords_np[:, 2]
    )
    sort_idx = np.argsort(keys_active, kind="stable")
    sorted_keys = keys_active[sort_idx]

    # Bounds-check the neighbor coords; clamp before hashing so OOB rows get a
    # sentinel that we then mask out (matches build_neighbor_table).
    in_bounds = ((connected_voxel >= 0) & (connected_voxel < grid_size)).all(axis=-1)
    clamped = np.clip(connected_voxel, 0, grid_size - 1)
    cv_keys = (
        clamped[..., 0] * (grid_size * grid_size) + clamped[..., 1] * grid_size + clamped[..., 2]
    )  # [M, 4]
    flat_keys = cv_keys.reshape(-1)
    positions = np.searchsorted(sorted_keys, flat_keys)
    positions = np.clip(positions, 0, n_active - 1)
    matched = sorted_keys[positions] == flat_keys
    indices = sort_idx[positions]
    valid_per_voxel = (matched & in_bounds.reshape(-1)).reshape(m_edges, 4)
    quad_indices_full = np.where(valid_per_voxel, indices.reshape(m_edges, 4), -1)

    # Keep only quads where all 4 ring voxels are active.
    valid_quads = (quad_indices_full >= 0).all(axis=-1)
    quad_indices = quad_indices_full[valid_quads].astype(np.int64)  # [L_q, 4]
    n_quads = quad_indices.shape[0]
    if n_quads == 0:
        return mx.array(mesh_vertices), mx.zeros((0, 3), dtype=mx.int32)

    # 4. γ-based diagonal selection per quad.
    gamma_per_quad = gamma_np[quad_indices]  # [L_q, 4]
    score_02 = gamma_per_quad[:, 0] * gamma_per_quad[:, 2]
    score_13 = gamma_per_quad[:, 1] * gamma_per_quad[:, 3]
    use_diag_02 = score_02 > score_13  # [L_q]

    # 5. Build triangles. Two candidate triangulations per quad, then where().
    tris_02 = quad_indices[:, _QUAD_SPLIT_DIAG_02].reshape(n_quads, 2, 3)
    tris_13 = quad_indices[:, _QUAD_SPLIT_DIAG_13].reshape(n_quads, 2, 3)
    mesh_triangles = np.where(use_diag_02[:, None, None], tris_02, tris_13).reshape(-1, 3)

    return mx.array(mesh_vertices), mx.array(mesh_triangles.astype(np.int32))
