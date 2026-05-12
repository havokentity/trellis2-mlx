"""SubMConv3 — parity tests against a numpy brute-force reference.

The brute-force reference walks each active voxel and each of the 27 kernel
positions explicitly, multiplying gathered input features by the matching
kernel slice and accumulating into the output. This is the spec's
``y_i = Σ_k W_k · x_{N(i,k)}`` definition rendered as a triple loop —
unambiguously correct, slow only at test scale.

The :func:`submconv3` implementation under test uses a "Masked Implicit
GEMM" (one big matmul over the gathered tensor) and must agree with the
reference to fp32 precision on small inputs and to mid-1e-4 on a realistic
9.6K-voxel × 128-channel workload (single-precision matmul drift).

Tests:

* ``test_submconv3_brute_force_parity_tiny`` — 8 voxels, 3 channels, hand-
  verifiable.
* ``test_submconv3_brute_force_parity_random`` — 256 voxels on a 16³ grid,
  parameterized over (C_in, C_out).
* ``test_submconv3_autograd_smoke`` — grads flow through the composite
  ``take + reshape + matmul`` op without a custom VJP (no Metal kernel yet).
* ``test_submconv3_perf_sanity`` — 9.6K voxels, 128 channels, marked
  ``slow``; just records wall-time so future regressions are visible.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from trellis2_mlx.nn.sparse_conv import SubMConv3, submconv3
from trellis2_mlx.ovoxel.data import build_neighbor_table


def _brute_force_submconv3(
    x: np.ndarray,
    weight: np.ndarray,
    neighbor_table: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Triple-loop reference for SubMConv3.

    For each output voxel i and kernel position k, look up the neighbor j
    in ``neighbor_table[i, k]``. If valid (>=0), accumulate ``W[k] @ x[j]``
    into ``y[i]``; otherwise skip.
    """
    n_active, c_in = x.shape
    _, _, c_out = weight.shape
    y = np.zeros((n_active, c_out), dtype=x.dtype)
    for i in range(n_active):
        for k in range(27):
            j = int(neighbor_table[i, k])
            if j < 0:
                continue
            y[i] += x[j] @ weight[k]
    if bias is not None:
        y = y + bias
    return y


def _random_active_set(n_active: int, resolution: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = rng.choice(resolution**3, size=n_active, replace=False)
    z = flat // (resolution * resolution)
    rem = flat % (resolution * resolution)
    y = rem // resolution
    x = rem % resolution
    return np.stack([z, y, x], axis=-1).astype(np.int32)


def test_submconv3_brute_force_parity_tiny() -> None:
    """2×2×2 corner cube, 3 in-channels, 4 out-channels — fully hand-verifiable."""
    coords = np.array(
        [(z, y, x) for z in (0, 1) for y in (0, 1) for x in (0, 1)],
        dtype=np.int32,
    )
    n_active = coords.shape[0]
    c_in, c_out = 3, 4
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_active, c_in)).astype(np.float32)
    w = rng.standard_normal((27, c_in, c_out)).astype(np.float32) * 0.1
    b = rng.standard_normal(c_out).astype(np.float32) * 0.1

    nt = np.asarray(build_neighbor_table(mx.array(coords), resolution=4))
    expected = _brute_force_submconv3(x, w, nt, b)

    actual = np.asarray(submconv3(mx.array(x), mx.array(w), mx.array(nt), mx.array(b)))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("c_in,c_out", [(8, 8), (16, 32), (64, 16)])
def test_submconv3_brute_force_parity_random(c_in: int, c_out: int) -> None:
    """256 random active voxels on a 16³ grid; varies (C_in, C_out)."""
    resolution = 16
    n_active = 256
    coords = _random_active_set(n_active, resolution, seed=c_in + c_out)
    rng = np.random.default_rng(c_in + c_out)
    x = rng.standard_normal((n_active, c_in)).astype(np.float32) * 0.5
    w = rng.standard_normal((27, c_in, c_out)).astype(np.float32) * 0.1

    nt = np.asarray(build_neighbor_table(mx.array(coords), resolution=resolution))
    expected = _brute_force_submconv3(x, w, nt)

    actual = np.asarray(submconv3(mx.array(x), mx.array(w), mx.array(nt)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_submconv3_autograd_smoke() -> None:
    """Gradients flow through the composite gather+matmul op without a custom VJP."""
    resolution = 8
    n_active = 32
    c_in, c_out = 4, 6
    coords = _random_active_set(n_active, resolution, seed=7)
    nt = build_neighbor_table(mx.array(coords), resolution=resolution)
    rng = np.random.default_rng(7)
    x_init = rng.standard_normal((n_active, c_in)).astype(np.float32) * 0.5

    layer = SubMConv3(c_in, c_out, bias=True)

    def loss_fn(weight: mx.array, bias: mx.array, x: mx.array) -> mx.array:
        return submconv3(x, weight, nt, bias).sum()

    grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
    g_w, g_b, g_x = grad_fn(layer.weight, layer.bias, mx.array(x_init))
    mx.eval(g_w, g_b, g_x)

    assert g_w.shape == layer.weight.shape
    assert g_b.shape == layer.bias.shape
    assert g_x.shape == (n_active, c_in)
    # Sum-of-output gradient w.r.t. bias is just count-per-channel; nonzero.
    assert float(mx.abs(g_b).sum()) > 0
    assert float(mx.abs(g_w).sum()) > 0
    assert float(mx.abs(g_x).sum()) > 0


@pytest.mark.slow
def test_submconv3_perf_sanity() -> None:
    """L≈9.6K, C_in=C_out=128 — realistic SLAT-stage shape. Records wall time."""
    resolution = 64
    n_active = 9600
    c_in = c_out = 128
    coords = _random_active_set(n_active, resolution, seed=0)
    nt = build_neighbor_table(mx.array(coords), resolution=resolution)
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((n_active, c_in)).astype(np.float32) * 0.5)
    w = mx.array(rng.standard_normal((27, c_in, c_out)).astype(np.float32) * 0.05)

    # Warm up
    y = submconv3(x, w, nt)
    mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(10):
        y = submconv3(x, w, nt)
        mx.eval(y)
    elapsed = (time.perf_counter() - t0) / 10

    # 9600 * 27 * 128 * 128 mul-adds = 4.25 GFLOPS per call
    gflops = 2 * n_active * 27 * c_in * c_out / 1e9
    rate = gflops / elapsed
    print(f"\n  SubMConv3 L={n_active} C={c_in}: {elapsed * 1e3:.2f} ms ({rate:.1f} GFLOP/s)")
    # No hard assertion — this test is informational, marked slow.
