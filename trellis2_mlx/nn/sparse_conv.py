"""Submanifold sparse 3D convolution (SubMConv3).

Implements ``PHASE0_SPEC.md §5.2``. The output voxel set equals the input
voxel set (no spatial expansion). Forward computes
``y_i = Σ_k W_k · x_{N(i, k)}`` for valid neighbors ``N(i, k)``; missing
neighbors (encoded ``-1`` in the neighbor table) contribute zero.

The current implementation is **MLX-native** — a gather + single implicit-GEMM
matmul, following the spec's "Masked Implicit GEMM" strategy. Because every
op (``take``, ``reshape``, ``matmul``) is differentiable in MLX, autograd
works out of the box; no custom VJP is needed until we replace the inner
loop with a Metal kernel (Phase 1 step 11).

The Metal-kernel replacement lives in
``trellis2_mlx/metal/kernels/sparse_conv_{fwd,bwd}.metal`` and will share
this module's ``nn.Module`` interface.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def submconv3(
    x: mx.array,
    weight: mx.array,
    neighbor_table: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """Apply a 3×3×3 submanifold sparse conv.

    Parameters
    ----------
    x : mx.array
        ``[L, C_in]`` per-voxel features.
    weight : mx.array
        ``[27, C_in, C_out]`` conv weights. Kernel positions are indexed in
        z-y-x scan order matching
        :func:`trellis2_mlx.ovoxel.data.build_neighbor_table` — slot 13 is
        the centre (self) tap.
    neighbor_table : mx.array
        ``[L, 27]`` int32 table from
        :func:`trellis2_mlx.ovoxel.data.build_neighbor_table`. Missing
        neighbors are encoded as ``-1``.
    bias : mx.array or None
        Optional ``[C_out]`` output bias.

    Returns
    -------
    mx.array
        ``[L, C_out]`` output features.

    Notes
    -----
    Implementation = **per-kernel-position accumulation**:

    1. Pad ``x`` with a trailing zero row so missing-neighbor index ``L``
       contributes zero.
    2. For each kernel position ``k ∈ [0, 27)``: gather rows of ``x_padded``
       indexed by ``neighbor_table[:, k]``, multiply by ``weight[k]``,
       accumulate into the running output.

    This is mathematically equivalent to the "Masked Implicit GEMM"
    formulation (one big matmul against a ``[27 * C_in, C_out]`` weight)
    but **bounds peak GPU memory** to a single ``[L, max(C_in, C_out)]``
    tensor per iteration. The fused implicit-GEMM blew past the Metal
    86 GB single-buffer cap at large ``L`` (~ 1M voxels) and small ``C``
    in the upsample stages of the SC-VAE decoder, where MLX's matmul
    appears to allocate intermediate scratch proportional to ``L * 27 *
    C_in * C_out``. The loop version keeps peak under a few GB.

    The Metal-kernel replacement (``sparse_conv_fwd.metal``) lands at
    Phase 1 step 5 / 11 — it will fuse the gather and the matmul into
    a single dispatch with the masked-implicit-GEMM strategy.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [L, C_in], got shape {tuple(x.shape)}")
    if weight.ndim != 3 or weight.shape[0] != 27:
        raise ValueError(f"weight must be [27, C_in, C_out], got shape {tuple(weight.shape)}")
    if neighbor_table.ndim != 2 or neighbor_table.shape[1] != 27:
        raise ValueError(f"neighbor_table must be [L, 27], got shape {tuple(neighbor_table.shape)}")

    n_active, c_in = x.shape
    _, weight_c_in, c_out = weight.shape
    if c_in != weight_c_in:
        raise ValueError(f"x channels ({c_in}) and weight in-channels ({weight_c_in}) must match")
    if neighbor_table.shape[0] != n_active:
        raise ValueError(
            f"neighbor_table.shape[0] ({neighbor_table.shape[0]}) must equal "
            f"x.shape[0] ({n_active})"
        )

    # Pad x with a zero row at index L; remap -1 → L so each gather pulls zeros.
    pad_idx = mx.array(n_active, dtype=neighbor_table.dtype)
    x_padded = mx.concatenate([x, mx.zeros((1, c_in), dtype=x.dtype)], axis=0)

    # Per-kernel-position accumulation. We start with bias (or zeros) so the
    # output is built up by 27 small matmuls of [L, C_in] @ [C_in, C_out]
    # rather than one big [L, 27*C_in] @ [27*C_in, C_out].
    out = (
        mx.zeros((n_active, c_out), dtype=x.dtype)
        if bias is None
        else mx.broadcast_to(bias.astype(x.dtype), (n_active, c_out)).astype(x.dtype)
    )
    for k in range(27):
        col = mx.where(neighbor_table[:, k] < 0, pad_idx, neighbor_table[:, k])
        gathered = mx.take(x_padded, col, axis=0)  # [L, C_in]
        out = out + gathered @ weight[k]  # [L, C_out]
    return out


class SubMConv3(nn.Module):
    """3×3×3 submanifold sparse convolution module.

    Holds the learned weight and (optional) bias. The neighbor table is
    passed in at call time — it depends on the active voxel set, not on
    the layer's parameters, so the same module instance can be reused
    across different active sets in a single forward pass.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    bias : bool
        Whether to allocate a per-output-channel bias parameter.
    """

    weight: mx.array
    bias: mx.array | None

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias
        # Kaiming-uniform style init: weight ~ U(-k, k) with k = sqrt(1 / (27 * fan_in)).
        # 27 because the receptive field is 3³ taps; matches PyTorch nn.Conv3d's default.
        k = (1.0 / (27 * in_channels)) ** 0.5
        self.weight = mx.random.uniform(low=-k, high=k, shape=(27, in_channels, out_channels))
        self.bias = mx.random.uniform(low=-k, high=k, shape=(out_channels,)) if bias else None

    def __call__(self, x: mx.array, neighbor_table: mx.array) -> mx.array:
        return submconv3(x, self.weight, neighbor_table, self.bias)
