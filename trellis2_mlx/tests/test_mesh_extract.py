"""Mesh extraction tests + the first end-to-end "random latent → GLB" demo.

The mesh-extraction algorithm itself (``extract_mesh``) is tested at the
primitive level: a hand-built 4-voxel ring that's known to produce one
quad (two triangles), and an empty-active-edges case.

The interesting test is ``test_shape_decoder_to_glb`` — it loads the real
SC-VAE shape decoder, feeds it a random latent on a tiny coarse grid, runs
mesh extraction on the output, and writes a GLB to a temporary location.
This is the first end-to-end "input goes in, 3D file comes out" path in
the project. Marked ``slow`` because of the 474M-param load.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from trellis2_mlx.models.vae import ShapeDecoder
from trellis2_mlx.ovoxel.mesh_extract import extract_mesh
from trellis2_mlx.ovoxel.postprocess import export_glb
from trellis2_mlx.tests.test_shape_decoder import _safetensors_load_all

_SHAPE_DEC = Path("reference/weights/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")


# ── Unit tests for extract_mesh ──────────────────────────────────────────


def test_extract_mesh_hand_built_one_quad() -> None:
    """4 voxels arranged so axis-0 δ on voxel 0 produces exactly one valid quad.

    Axis-0 ring offsets are ``{(0,0,0), (0,0,1), (0,1,1), (0,1,0)}``. We
    place 4 active voxels at exactly those coords from voxel 0 at the origin,
    set ``δ[0, 0] = True``, and verify the extractor emits 2 triangles
    referring to voxels {0, 1, 2, 3} in the right slot order.
    """
    coords = mx.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        dtype=mx.int32,
    )
    v = mx.array(
        [[0.5, 0.5, 0.5]] * 4,
        dtype=mx.float32,
    )  # centre dual vertex in each voxel
    delta_logits = mx.array(
        [[1.0, -1.0, -1.0]] + [[-1.0, -1.0, -1.0]] * 3,  # axis-0 on voxel 0 only
        dtype=mx.float32,
    )
    gamma = mx.array([[1.0], [1.0], [1.0], [1.0]], dtype=mx.float32)

    verts, faces = extract_mesh(coords, v, delta_logits, gamma, grid_size=2)
    verts_np = np.asarray(verts)
    faces_np = np.asarray(faces)

    # 4 dual vertices, 2 triangles
    assert verts_np.shape == (4, 3)
    assert faces_np.shape == (2, 3)
    # All face indices reference the 4 voxels we placed
    assert set(faces_np.flatten().tolist()) == {0, 1, 2, 3}


def test_extract_mesh_no_active_edges() -> None:
    """δ all negative → no faces emitted."""
    coords = mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.int32)
    v = mx.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=mx.float32)
    delta_logits = mx.array([[-1.0] * 3, [-1.0] * 3], dtype=mx.float32)
    gamma = mx.array([[1.0], [1.0]], dtype=mx.float32)

    verts, faces = extract_mesh(coords, v, delta_logits, gamma, grid_size=4)
    assert verts.shape == (2, 3)
    assert faces.shape == (0, 3)


def test_extract_mesh_delta_with_missing_ring_voxel() -> None:
    """δ True but the ring is incomplete → no faces for that edge."""
    # Only 3 voxels of the 4-voxel ring; the 4th (0,1,0) is missing.
    coords = mx.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=mx.int32)
    v = mx.array([[0.5, 0.5, 0.5]] * 3, dtype=mx.float32)
    delta_logits = mx.array([[1.0, -1.0, -1.0]] * 3, dtype=mx.float32)
    gamma = mx.array([[1.0]] * 3, dtype=mx.float32)

    verts, faces = extract_mesh(coords, v, delta_logits, gamma, grid_size=4)
    assert verts.shape == (3, 3)
    assert faces.shape == (0, 3)


def test_extract_mesh_gamma_diagonal_selection() -> None:
    """γ values bias the chosen diagonal.

    Quad slots 0 and 2 are opposite corners; slots 1 and 3 are the other
    pair. ``γ[0]·γ[2] > γ[1]·γ[3]`` selects diagonal 0-2 → triangles
    (0,1,2) and (0,2,3). Otherwise diagonal 1-3 → triangles (0,1,3) and
    (3,1,2).
    """
    coords = mx.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        dtype=mx.int32,
    )
    v = mx.array([[0.5, 0.5, 0.5]] * 4, dtype=mx.float32)
    delta_logits = mx.array(
        [[1.0, -1.0, -1.0]] + [[-1.0, -1.0, -1.0]] * 3,
        dtype=mx.float32,
    )
    # Bias toward diagonal 1-3
    gamma = mx.array([[0.1], [10.0], [0.1], [10.0]], dtype=mx.float32)
    _, faces = extract_mesh(coords, v, delta_logits, gamma, grid_size=2)
    faces_np = np.asarray(faces)
    # diagonal 1-3 → (0,1,3) and (3,1,2)
    expected = {(0, 1, 3), (3, 1, 2)}
    got = {tuple(t.tolist()) for t in faces_np}
    assert got == expected, f"expected {expected}, got {got}"


# ── End-to-end: ShapeDecoder → mesh → GLB ────────────────────────────────


@pytest.mark.slow
@pytest.mark.reference
def test_shape_decoder_to_glb(tmp_path: Path) -> None:
    """First end-to-end pipeline: random latent → SC-VAE shape decoder →
    Flexible Dual Grid mesh extraction → GLB file.

    The latent is random noise, so the resulting mesh is essentially
    garbage geometrically — but it IS a real 3D mesh that any glTF
    viewer can open, and that proves the *whole* shape-decoding path
    works.
    """
    if not _SHAPE_DEC.exists():
        pytest.skip(f"shape decoder weights not found at {_SHAPE_DEC}")
    from trellis2_mlx.utils.weight_convert import shape_decoder_from_pt_state_dict

    decoder = ShapeDecoder()
    state = _safetensors_load_all(_SHAPE_DEC)
    decoder.load_weights(shape_decoder_from_pt_state_dict(state))

    rng = np.random.default_rng(0)
    # Tiny coarse grid: 8 active voxels at 4³ → 64³ output (4 × 2⁴).
    coarse_res = 4
    n_coarse = 8
    flat = rng.choice(coarse_res**3, size=n_coarse, replace=False)
    z = flat // (coarse_res**2)
    rem = flat % (coarse_res**2)
    y = rem // coarse_res
    x = rem % coarse_res
    coords = mx.array(np.stack([z, y, x], axis=-1).astype(np.int32))
    latent = mx.array(rng.standard_normal((n_coarse, 32)).astype(np.float32) * 0.5)

    out = decoder(latent, coords, coarse_resolution=coarse_res)
    verts, faces = extract_mesh(
        out.coords, out.v, out.delta_logits, out.gamma, grid_size=out.output_resolution
    )

    print(
        f"\n  end-to-end OK: {n_coarse} latent voxels @ {coarse_res}³ → "
        f"{verts.shape[0]} dual vertices, {faces.shape[0]} triangles @ "
        f"{out.output_resolution}³"
    )

    # Sanity: vertices in the aabb (default [-0.5, 0.5]); faces reference
    # only valid vertex indices.
    verts_np = np.asarray(verts)
    faces_np = np.asarray(faces)
    assert verts_np.min() > -0.6 and verts_np.max() < 0.6
    if faces_np.size:
        assert faces_np.min() >= 0
        assert faces_np.max() < verts_np.shape[0]

    # Write a GLB and confirm trimesh can read it back.
    out_path = tmp_path / "shape_decoder_smoke.glb"
    written = export_glb(verts, faces, out_path)
    assert written.exists()
    assert written.stat().st_size > 100  # GLB header alone is ~80 bytes; we want geometry too

    import trimesh

    loaded = trimesh.load(str(written), force="mesh")
    # Trimesh strips unreferenced vertices on load; just check faces survived.
    assert loaded.faces.shape == faces_np.shape
    print(f"  wrote {written} ({written.stat().st_size / 1024:.1f} KB)")
