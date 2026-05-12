"""Rectified-flow sampler with CFG, CFG-rescale, and guidance interval.

Implements the inference path from
``reference/microsoft-trellis2/trellis2/pipelines/samplers/flow_euler.py``
plus the two mixins
(``classifier_free_guidance_mixin.py``, ``guidance_interval_mixin.py``).

The sampler is the **same** for all three TRELLIS DiT stages; only the
``params`` dict from ``pipeline.json`` differs:

============= ======= ============= ============== =================== =========
stage         steps   strength      rescale        interval            rescale_t
============= ======= ============= ============== =================== =========
SS (1)        12      7.5           0.7            (0.6, 1.0)          5.0
shape SLAT(2) 12      7.5           0.5            (0.6, 1.0)          3.0
texture(3)    12      1.0           0.0            (0.6, 0.9)          3.0
============= ======= ============= ============== =================== =========

API summary
-----------

* :class:`RectifiedFlowSampler` — Euler integration of the v-field with
  ``sigma_min`` parameterization.
* ``sample(model_fn, noise, cond, neg_cond, ...)`` returns the denoised
  sample. ``model_fn(x, t_scaled, cond)`` is the user's per-step callable —
  it returns the predicted velocity. ``t_scaled`` is in ``[0, 1000]``
  (matches ``flow_euler.py:45``: ``t_tensor = torch.tensor([1000 * t] * B)``).

Math
----

Conventions (matches upstream ``_pred_to_xstart`` / ``_xstart_to_pred`` /
``sample_once``):

* ``x_0 = (1 - σ_min) · x_t - (σ_min + (1 - σ_min) · t) · v``
* ``v   = ((1 - σ_min) · x_t - x_0) / (σ_min + (1 - σ_min) · t)``
* Euler step: ``x_{t-Δt} = x_t - (t - t_prev) · v``

CFG combination (matches ``classifier_free_guidance_mixin.py:17``):

  ``v = guidance_strength · v_cond + (1 - guidance_strength) · v_uncond``

CFG rescale (Lin et al. — ``classifier_free_guidance_mixin.py:20-27``):

  ``x0_cfg     = pred_to_xstart(x_t, t, v_cfg)``
  ``x0_pos     = pred_to_xstart(x_t, t, v_pos)``
  ``x0_rescale = x0_cfg · std(x0_pos) / std(x0_cfg)``
  ``x0_final   = guidance_rescale · x0_rescale + (1 - guidance_rescale) · x0_cfg``
  ``v_final    = xstart_to_pred(x_t, t, x0_final)``

Guidance interval (``guidance_interval_mixin.py:10``):

  When ``t ∉ [interval[0], interval[1]]``, skip CFG entirely (use the
  conditional ``v_cond`` directly).

t-schedule rescale (``flow_euler.py:115-117``):

  ``t_rescaled = rescale_t · t / (1 + (rescale_t - 1) · t)``

For ``rescale_t = 1`` this is identity. ``rescale_t > 1`` concentrates
the step density near ``t = 1`` (the noisy end of the schedule).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class SamplerParams:
    """Per-stage CFG / schedule parameters from ``pipeline.json``."""

    steps: int = 12
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.5
    guidance_interval: tuple[float, float] = (0.6, 1.0)
    rescale_t: float = 3.0


class RectifiedFlowSampler:
    """Euler integration of the rectified-flow velocity field.

    Wraps a ``model_fn`` that produces ``v_θ(x_t, t, cond)`` at each step.
    ``sigma_min`` is the lower bound of the integration schedule (default
    ``1e-5`` matches every published checkpoint config).

    The sampler does *not* know about the DiT internals — pass any
    callable that takes ``(x, t_scaled_to_1000, cond, **kwargs)`` and
    returns a velocity-shaped tensor. ``kwargs`` are forwarded to
    ``model_fn`` per step (e.g. ``coords=...`` for the SLAT DiTs).
    """

    def __init__(self, sigma_min: float = 1e-5) -> None:
        self.sigma_min = sigma_min

    # ── velocity / xstart helpers (match flow_euler.py) ────────────────

    def _pred_to_xstart(self, x_t: mx.array, t: float, v: mx.array) -> mx.array:
        return (1.0 - self.sigma_min) * x_t - (self.sigma_min + (1.0 - self.sigma_min) * t) * v

    def _xstart_to_pred(self, x_t: mx.array, t: float, x_0: mx.array) -> mx.array:
        return ((1.0 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1.0 - self.sigma_min) * t)

    # ── single-step v prediction (CFG + rescale + interval) ────────────

    def _predict_velocity(
        self,
        model_fn: Callable[..., mx.array],
        x_t: mx.array,
        t: float,
        cond: object,
        neg_cond: object | None,
        *,
        guidance_strength: float,
        guidance_rescale: float,
        guidance_interval: tuple[float, float],
        **kwargs: object,
    ) -> mx.array:
        """Predict ``v`` at ``(x_t, t)`` with full CFG + rescale + interval logic."""
        t_scaled = mx.array([1000.0 * t], dtype=mx.float32)

        # Guidance interval — outside the interval, use conditional only.
        in_interval = guidance_interval[0] <= t <= guidance_interval[1]
        if not in_interval or guidance_strength == 1.0 or neg_cond is None:
            return model_fn(x_t, t_scaled, cond, **kwargs)
        if guidance_strength == 0.0:
            return model_fn(x_t, t_scaled, neg_cond, **kwargs)

        # CFG combine
        v_pos = model_fn(x_t, t_scaled, cond, **kwargs)
        v_neg = model_fn(x_t, t_scaled, neg_cond, **kwargs)
        v_cfg = guidance_strength * v_pos + (1.0 - guidance_strength) * v_neg

        if guidance_rescale <= 0.0:
            return v_cfg

        # CFG rescale (Lin et al.). Compute x0 std on both v_pos and v_cfg,
        # rescale v_cfg's implied x0 to match v_pos's std, then convert back.
        x0_pos = self._pred_to_xstart(x_t, t, v_pos)
        x0_cfg = self._pred_to_xstart(x_t, t, v_cfg)
        # std over all dims except batch (axis 0); the SLAT DiT runs with
        # no batch dim, so std over all dims is the right thing.
        if x0_pos.ndim == 1:
            std_pos = mx.std(x0_pos, keepdims=True)
            std_cfg = mx.std(x0_cfg, keepdims=True)
        else:
            # std over flattened tail dims, keep batch
            std_pos = mx.std(x0_pos.reshape(x0_pos.shape[0], -1), axis=1, keepdims=True)
            std_cfg = mx.std(x0_cfg.reshape(x0_cfg.shape[0], -1), axis=1, keepdims=True)
            # Reshape so it broadcasts with the original (N-dim) tensors
            for _ in range(x0_pos.ndim - 2):
                std_pos = std_pos[..., None]
                std_cfg = std_cfg[..., None]
        # Avoid division by zero on degenerate inputs.
        x0_rescale = x0_cfg * (std_pos / (std_cfg + 1e-12))
        x0_final = guidance_rescale * x0_rescale + (1.0 - guidance_rescale) * x0_cfg
        return self._xstart_to_pred(x_t, t, x0_final)

    # ── full integration ───────────────────────────────────────────────

    @staticmethod
    def t_schedule(steps: int, rescale_t: float) -> list[float]:
        """Build the linspace + reparametrize schedule from ``flow_euler.py:115-117``."""
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        return t_seq.tolist()

    def sample(
        self,
        model_fn: Callable[..., mx.array],
        noise: mx.array,
        cond: object,
        neg_cond: object | None = None,
        *,
        params: SamplerParams | None = None,
        **model_kwargs: object,
    ) -> mx.array:
        """Sample ``x_0`` from ``noise`` by Euler-integrating the v-field.

        Parameters
        ----------
        model_fn : callable
            ``model_fn(x_t, t_scaled, cond_arg, **kwargs) → v`` where
            ``cond_arg`` is either ``cond`` or ``neg_cond`` per CFG branch
            and ``t_scaled`` is a ``[1]`` array in ``[0, 1000]``.
        noise : mx.array
            Initial sample at ``t = 1`` (pure noise).
        cond : object
            Conditional signal to forward into ``model_fn`` (e.g. DINOv3
            features for the SLAT DiTs).
        neg_cond : object or None
            Unconditional signal. The pipeline uses
            ``torch.zeros_like(cond)``; pass ``None`` here to disable CFG
            entirely.
        params : SamplerParams or None
            Per-stage schedule + CFG parameters.
        **model_kwargs : object
            Forwarded to every ``model_fn`` call (e.g. ``coords=...``).

        Returns
        -------
        mx.array
            Final denoised sample. Same shape and dtype as ``noise``.
        """
        params = params or SamplerParams()
        sample = noise
        t_seq = self.t_schedule(params.steps, params.rescale_t)
        for t, t_prev in zip(t_seq[:-1], t_seq[1:], strict=False):
            v = self._predict_velocity(
                model_fn,
                sample,
                t,
                cond,
                neg_cond,
                guidance_strength=params.guidance_strength,
                guidance_rescale=params.guidance_rescale,
                guidance_interval=params.guidance_interval,
                **model_kwargs,
            )
            sample = sample - (t - t_prev) * v
        return sample


__all__ = ["RectifiedFlowSampler", "SamplerParams"]
