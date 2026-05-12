"""MLX bindings for the custom Metal kernels.

Strategy (per ``PHASE0_SPEC.md §6.2``):

* Forward-only ops (neighbor build, prefix sum, mesh extraction, trilinear bake)
  go through ``mx.fast.metal_kernel(...)``.
* Ops that need autograd (sparse attention, sparse conv) go through the MLX
  C++ extension path so we can register a custom VJP.

Kernel source lives in ``trellis2_mlx/metal/kernels/*.metal``. Compiled
``.metallib`` artifacts are emitted under ``build/`` (gitignored).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

KERNELS_DIR = Path(__file__).parent / "kernels"


# ── Forward-only ops ────────────────────────────────────────────────────────


def neighbor_build(coords: mx.array) -> mx.array:
    """Build a ``[L, 27]`` int32 neighbor table. See spec §5.3."""
    raise NotImplementedError


def prefix_sum(mask: mx.array) -> mx.array:
    """Exclusive parallel prefix-sum over a flat mask; used for compaction. See spec §5.5."""
    raise NotImplementedError


def mesh_extract(
    coords: mx.array,
    v: mx.array,
    delta: mx.array,
    gamma: mx.array,
    resolution: int,
) -> tuple[mx.array, mx.array]:
    """Flexible Dual Grid mesh extraction. See spec §5.6."""
    raise NotImplementedError


def trilinear_bake(
    coords: mx.array,
    features: mx.array,
    query_points: mx.array,
    resolution: int,
) -> mx.array:
    """Trilinear interpolation of per-voxel features at query points. See spec §5.7."""
    raise NotImplementedError


# ── Ops with autograd (registered via MLX C++ extension) ────────────────────


def sparse_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """Flash-style sparse-token self-attention with VJP. See spec §5.1."""
    raise NotImplementedError


def submanifold_conv3(
    x: mx.array,
    weight: mx.array,
    neighbor_table: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """3×3×3 submanifold sparse conv with VJP. See spec §5.2."""
    raise NotImplementedError
