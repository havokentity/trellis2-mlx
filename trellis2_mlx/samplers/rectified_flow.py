"""Rectified-flow sampler with classifier-free guidance.

Implements ``PHASE0_SPEC.md §2.1``. Forward process is ``x(t) = (1−t)x₀ + tε``;
inference integrates the predicted velocity field ``v_θ`` from ``t=1`` (noise)
to ``t=0`` (data). Default 25–50 steps per stage with CFG scale ~3–7.5 —
exact defaults pending §8 Q4.

Reference: Liu et al., "Flow Straight and Fast", arXiv 2209.03003.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


def sample(
    velocity_fn: Callable[..., mx.array],
    x_init: mx.array,
    *,
    num_steps: int = 25,
    cfg_scale: float = 3.0,
    cond: object | None = None,
    uncond: object | None = None,
) -> mx.array:
    """Integrate the rectified-flow ODE from noise to data.

    Parameters
    ----------
    velocity_fn : callable
        ``velocity_fn(x, t, cond)`` returning the predicted velocity at ``(x, t)``.
    x_init : mx.array
        Initial noise sample at ``t = 1``.
    num_steps : int
        Number of integration steps (uniform schedule over ``t ∈ [0, 1]``).
    cfg_scale : float
        Classifier-free guidance scale. Combined as
        ``v = v_uncond + s * (v_cond - v_uncond)``.
    cond, uncond : object
        Conditioning / unconditional inputs passed through to ``velocity_fn``.
    """
    raise NotImplementedError("rectified-flow sampler lands in Phase 1 step 8")
