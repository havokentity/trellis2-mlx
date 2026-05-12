"""3D rotary positional embeddings for sparse-voxel attention.

Implements ``PHASE0_SPEC.md §5.9`` with the corrections from
``docs/open-questions-resolved.md`` Q1.

Recipe (matches ``reference/microsoft-trellis2/trellis2/modules/sparse/attention/rope.py``):

* ``freq_dim = head_dim // 2 // 3``  (= 21 for head_dim=128)
* ``inv_freq[k] = 1 / base ** (k / freq_dim)``  with ``base = 10000``
* For each voxel ``i``, compute ``angles[i, axis, k] = coord[i, axis] * inv_freq[k]``
  with ``axis ∈ {z, y, x}`` matching the coord-row order.
* Flatten ``angles`` to ``[N, 3 * freq_dim]`` (per-axis contiguous blocks,
  **not** interleaved) and zero-pad to ``head_dim / 2 = 64``.
* Treat the feature tensor as complex pairs ``(channel[2k], channel[2k+1])``
  and multiply each pair by ``exp(i * angles[k])`` — i.e. the standard
  complex-multiply rotation, **not** the "rotate_half" variant DINOv3 uses.

The per-axis ordering and the zero-pad slot are what distinguish this
from the DINOv3 implementation in ``models/dinov3.py``; do not share code
between the two.
"""

from __future__ import annotations

import mlx.core as mx


def build_rope_3d_phases(
    coords: mx.array,
    head_dim: int,
    *,
    base: float = 10000.0,
) -> tuple[mx.array, mx.array]:
    """Compute the ``(cos, sin)`` phases for 3D RoPE.

    Parameters
    ----------
    coords : mx.array
        ``[N, 3]`` voxel coordinates (any int / float dtype).
    head_dim : int
        Attention head dimension. Must be even; ``head_dim // 2`` should
        normally be ``≥ 3 * (head_dim // 2 // 3)`` (always true).
    base : float
        Base frequency. ``10000`` for the published DiT checkpoints.

    Returns
    -------
    cos, sin : mx.array
        ``[N, head_dim // 2]`` real arrays. Trailing slots beyond
        ``3 * (head_dim // 2 // 3)`` are zero-angle (cos=1, sin=0).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even; got {head_dim}")
    pairs = head_dim // 2
    freq_dim = pairs // 3
    pad = pairs - 3 * freq_dim  # 1 when head_dim=128 (64 - 63 = 1)

    inv_freq = 1.0 / (base ** (mx.arange(freq_dim, dtype=mx.float32) / freq_dim))
    # coords: [N, 3]  →  angles: [N, 3, freq_dim]
    angles = coords[:, :, None].astype(mx.float32) * inv_freq[None, None, :]
    angles = angles.reshape(coords.shape[0], 3 * freq_dim)
    if pad > 0:
        angles = mx.concatenate([angles, mx.zeros((coords.shape[0], pad))], axis=-1)
    return mx.cos(angles), mx.sin(angles)


def apply_rope_3d(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the last dim of ``x`` by precomputed ``cos`` / ``sin`` phases.

    Parameters
    ----------
    x : mx.array
        ``[..., D]`` tensor — typically ``[N, H, D]`` for Q or K.
    cos, sin : mx.array
        ``[N, D // 2]`` phases from :func:`build_rope_3d_phases`. The shape
        broadcasts across any intermediate dims of ``x`` (e.g. the head
        dim) — caller need not unsqueeze.
    """
    d = x.shape[-1]
    pairs = d // 2
    # x as complex pairs along the last dim:
    # x[..., 2k], x[..., 2k+1]  →  real / imaginary slots of complex k.
    x_pairs = x.reshape(*x.shape[:-1], pairs, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    # Broadcast cos/sin into the same shape as x_real/imag. cos has shape
    # [N, pairs]; x_real has shape [N, ..., pairs]. Insert singleton dims
    # between N and pairs so broadcasting works for arbitrary middle axes.
    while cos.ndim < x_real.ndim:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    rotated_real = x_real * cos - x_imag * sin
    rotated_imag = x_real * sin + x_imag * cos
    out = mx.stack([rotated_real, rotated_imag], axis=-1).reshape(x.shape)
    return out.astype(x.dtype)


__all__ = ["apply_rope_3d", "build_rope_3d_phases"]
