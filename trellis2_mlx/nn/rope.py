"""3D rotary positional embeddings.

Implements ``PHASE0_SPEC.md §5.9``. Per token, splits the head dim into three
equal groups (one per spatial axis); within each group, applies a standard
2D rotation by ``θ_k = coord_axis * base^(-2k / (D/3))``. Applied to Q and K
*before* the attention dot-product (and before QK-Norm? — checked against
upstream, see ``docs/open-questions-resolved.md`` Q1).

The exact base frequency and axis ordering must match the checkpoint or
generation collapses — these are tracked as open question §8 Q1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


def apply_rope_3d(
    x: "mx.array",
    coords: "mx.array",
    *,
    base: float = 10000.0,
) -> "mx.array":
    """Rotate the last dim of ``x`` by 3D RoPE driven by ``coords``.

    Parameters
    ----------
    x : mx.array
        ``[..., D]`` token features (Q or K). ``D`` must be divisible by 6
        (3 axes × pair).
    coords : mx.array
        ``[..., 3]`` int32 voxel coordinates.
    base : float
        RoPE base frequency. Default 10000.0; pending upstream verification.

    Returns
    -------
    mx.array
        Rotated features, same shape and dtype as ``x``.
    """
    raise NotImplementedError("RoPE-3D lands with the attention module in Phase 1 step 7")
