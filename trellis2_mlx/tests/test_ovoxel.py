"""OVoxel + neighbor-table tests.

Two layers:

* ``test_neighbor_table_hand_built`` — a 2³ corner cube on a 4³ grid with a
  precomputed expected table. Verifies axis ordering (z-y-x scan), self-slot
  (index 13 = (0,0,0)), out-of-bounds → -1, and missing-active → -1 all in
  one ~20-voxel example.
* ``test_neighbor_table_brute_force_parity`` — generates a random ~5K active
  set on a 32³ grid and compares :func:`build_neighbor_table` against a
  hash-of-tuples brute-force reference. Catches any vectorization bug at
  realistic scale.
* ``test_neighbor_offset_index_round_trip`` — sanity check on the
  ``(dz, dy, dx) ↔ slot`` mapping.

The MLX implementation is currently numpy-backed (see module docstring of
``trellis2_mlx.ovoxel.data``); once the Metal kernel lands the tests stay
unchanged.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from trellis2_mlx.ovoxel.data import (
    _NEIGHBOR_OFFSETS,
    build_neighbor_table,
    child_to_parent_coords,
    neighbor_offset_index,
)


def _brute_force_neighbors(coords: np.ndarray, resolution: int) -> np.ndarray:
    """Reference impl: build a Python dict, scan 27 offsets per voxel."""
    table: dict[tuple[int, int, int], int] = {
        (int(z), int(y), int(x)): i for i, (z, y, x) in enumerate(coords)
    }
    out = np.full((coords.shape[0], 27), -1, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords):
        for k, (dz, dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
            nz, ny, nx = int(z + dz), int(y + dy), int(x + dx)
            if not (0 <= nz < resolution and 0 <= ny < resolution and 0 <= nx < resolution):
                continue
            out[i, k] = table.get((nz, ny, nx), -1)
    return out


def test_neighbor_offset_index_round_trip() -> None:
    for k, (dz, dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
        assert neighbor_offset_index(int(dz), int(dy), int(dx)) == k
    # Self slot
    assert neighbor_offset_index(0, 0, 0) == 13
    # Corner extremes
    assert neighbor_offset_index(-1, -1, -1) == 0
    assert neighbor_offset_index(1, 1, 1) == 26


def test_neighbor_table_hand_built() -> None:
    """A 2×2×2 cube of active voxels at the origin of a 4³ grid."""
    coords = np.array(
        [(z, y, x) for z in (0, 1) for y in (0, 1) for x in (0, 1)],
        dtype=np.int32,
    )
    table = np.asarray(build_neighbor_table(mx.array(coords), resolution=4))
    assert table.shape == (8, 27)

    # Voxel 0 = (0, 0, 0). Slot 13 = self.
    # Slot 26 = (+1, +1, +1) → coord (1, 1, 1) which is voxel index 7 (z=1, y=1, x=1 in our enumeration).
    self_idx = 0  # by construction
    assert table[self_idx, 13] == self_idx
    # The (+z, +y, +x) neighbor of voxel 0 is voxel 7 (since our enum is z-outermost, x-innermost).
    enum_idx = {(int(zz), int(yy), int(xx)): i for i, (zz, yy, xx) in enumerate(coords)}
    assert table[self_idx, neighbor_offset_index(1, 1, 1)] == enum_idx[(1, 1, 1)]
    # The (-z, -y, -x) neighbor of voxel 0 is out of bounds → -1.
    assert table[self_idx, neighbor_offset_index(-1, -1, -1)] == -1
    # The (+z, 0, 0) neighbor of voxel 0 is voxel (1, 0, 0).
    assert table[self_idx, neighbor_offset_index(1, 0, 0)] == enum_idx[(1, 0, 0)]
    # Voxel 7 = (1, 1, 1). Its (+1, 0, 0) neighbor would be (2, 1, 1), which is in-bounds
    # but NOT in the active set → -1.
    assert table[enum_idx[(1, 1, 1)], neighbor_offset_index(1, 0, 0)] == -1


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_neighbor_table_brute_force_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    resolution = 32
    n_total = resolution**3
    n_active = 5000
    flat = rng.choice(n_total, size=n_active, replace=False)
    z = flat // (resolution * resolution)
    rem = flat % (resolution * resolution)
    y = rem // resolution
    x = rem % resolution
    coords = np.stack([z, y, x], axis=-1).astype(np.int32)

    mlx_table = np.asarray(build_neighbor_table(mx.array(coords), resolution=resolution))
    ref_table = _brute_force_neighbors(coords, resolution)

    diff = np.where(mlx_table != ref_table)
    if diff[0].size:
        i0 = diff[0][0]
        k0 = diff[1][0]
        pytest.fail(
            f"mismatch at voxel {i0} coord {coords[i0]} slot {k0} offset {_NEIGHBOR_OFFSETS[k0]}: "
            f"mlx={mlx_table[i0, k0]} ref={ref_table[i0, k0]}"
        )


def test_neighbor_table_empty() -> None:
    table = build_neighbor_table(mx.array(np.zeros((0, 3), dtype=np.int32)), resolution=32)
    assert table.shape == (0, 27)


def test_neighbor_table_bounds_check_rejects_invalid() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 4\)"):
        build_neighbor_table(mx.array(np.array([[0, 0, 4]], dtype=np.int32)), resolution=4)


def test_child_to_parent_coords() -> None:
    fine = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [3, 5, 7],
            [10, 10, 10],
        ],
        dtype=np.int32,
    )
    parent = np.asarray(child_to_parent_coords(mx.array(fine)))
    expected = fine // 2
    assert np.array_equal(parent, expected)
