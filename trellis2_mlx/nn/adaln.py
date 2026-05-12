"""AdaLN-single modulation (PixArt-α style).

Implements ``PHASE0_SPEC.md §4.4`` / §5.8 with the ``share_mod=True`` variant
used by the published DiT checkpoints (see
``docs/open-questions-resolved.md``: per-block ``modulation`` Parameter +
shared ``adaLN_modulation`` MLP at model root).

Two pieces:

* :class:`TimestepEmbedder` — sinusoidal embedding + 2-layer MLP. Matches
  ``sparse_structure_flow.py:TimestepEmbedder`` (12-53).
* :class:`AdaLNSingle` — the shared ``SiLU + Linear(C, 6C)`` MLP that
  consumes the timestep embedding and produces a ``[B, 6*C]`` modulation
  vector. Each DiT block then adds its own learned ``[6*C]`` bias before
  chunking into ``(shift, scale, gate) × 2``.

Reference: Chen et al., "PixArt-α", arXiv 2310.00426.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def sinusoidal_timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """Sinusoidal positional embedding for diffusion timesteps.

    Matches ``trellis2/models/sparse_structure_flow.py:25-48``: half the
    dim is cos, half is sin, with a zero-padded tail if ``dim`` is odd.

    Parameters
    ----------
    t : mx.array
        ``[B]`` timestep values (any float dtype). The pipeline passes
        ``1000 * t`` here (see ``flow_euler.py:45``).
    dim : int
        Embedding dimension. ``frequency_embedding_size`` in upstream
        terms — typically 256.
    max_period : float
        Controls the minimum frequency.
    """
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None, :]  # [B, half]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((t.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal + MLP timestep embedder.

    Matches ``sparse_structure_flow.py:TimestepEmbedder`` (12-53). Output
    shape ``[B, hidden_size]``.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def __call__(self, t: mx.array) -> mx.array:
        emb = sinusoidal_timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb)


class AdaLNSingle(nn.Module):
    """Shared ``SiLU + Linear(C, 6C)`` modulation predictor (PixArt-α).

    Lives at the DiT model root when ``share_mod=True`` (always True for
    the published checkpoints). Each DiT block adds its own learned
    ``[6C]`` bias (see :class:`trellis2_mlx.nn.dit_block.ModulatedDiTCrossBlock`)
    before chunking into the six scalars
    ``(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        # The upstream stores this as ``nn.Sequential(SiLU, Linear(C, 6C))``
        # which puts the Linear at index 1. Our state-dict mapping in
        # weight_convert preserves the index naming.
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))

    def __call__(self, t_emb: mx.array) -> mx.array:
        return self.mlp(t_emb)


__all__ = ["AdaLNSingle", "TimestepEmbedder", "sinusoidal_timestep_embedding"]
