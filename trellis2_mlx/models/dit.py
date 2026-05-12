"""DiT flow models — sparse geometry / texture generators.

Implements ``PHASE0_SPEC.md §4.4`` for the **sparse** generators (stages 2 and
3 of the inference pipeline). Stage 1 (sparse-structure DiT) lives in
:class:`SparseStructureFlowModel` below — it operates on a *dense* small
grid (16³ in the checkpoint) rather than the sparse active voxels stages
2/3 see, so it gets its own class.

Architecture per ``configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json``
(SLAT stages):

* ``input_layer``        — ``Linear(in_channels → 1536)``
* ``t_embedder``         — sinusoidal timestep MLP
* ``adaLN_modulation``   — shared ``SiLU + Linear(1536, 9216)`` (``share_mod=True``)
* 30 × ``ModulatedDiTCrossBlock`` (12 heads × 128 head_dim, mlp_ratio 5.3334)
* parameter-free ``LayerNorm``
* ``out_layer``          — ``Linear(1536, out_channels)``

In channels:
* shape SLAT DiT  → 32  (32-dim latents from the SLAT bottleneck)
* texture SLAT DiT → 64 (32 shape + 32 noise, concatenated channel-wise)
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from trellis2_mlx.nn.adaln import AdaLNSingle, TimestepEmbedder
from trellis2_mlx.nn.dit_block import ModulatedDiTCrossBlock


@dataclass(frozen=True)
class SLatFlowConfig:
    """Static config for a sparse-latent DiT (stages 2 and 3).

    Defaults match every published SLAT DiT checkpoint. Override
    ``in_channels`` and ``resolution`` per variant (32-ch / 32³ or 64³ for
    shape, 64-ch / 32³ or 64³ for texture).
    """

    resolution: int = 32  # 32 for 512-output; 64 for 1024-output ft variants
    in_channels: int = 32  # 32 for shape; 64 for texture (concat shape + noise)
    out_channels: int = 32
    model_channels: int = 1536
    cond_channels: int = 1024
    num_blocks: int = 30
    num_heads: int = 12
    mlp_ratio: float = 5.3334
    rope_base: float = 10000.0


class SLatFlowModel(nn.Module):
    """Sparse-latent DiT — flow-matching generator on the SLAT active set.

    Forward pass (matches ``trellis2/models/structured_latent_flow.py:169-199``):

    1. ``h = input_layer(x)``                              # [L, 1536]
    2. ``t_emb = adaLN_modulation(t_embedder(t))``         # [B, 9216]
    3. For each block: ``h = block(h, coords, t_emb, cond)``
    4. ``h = F.layer_norm(h)`` (parameter-free)
    5. ``h = out_layer(h)``                                 # [L, out_channels]

    Note the absence of any positional-embedding step at the model level —
    3D RoPE is applied *inside* each block's self-attention against the
    voxel coords.

    Parameters
    ----------
    cfg : SLatFlowConfig | None
    """

    def __init__(self, cfg: SLatFlowConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SLatFlowConfig()
        c = self.cfg

        self.input_layer = nn.Linear(c.in_channels, c.model_channels, bias=True)
        self.t_embedder = TimestepEmbedder(c.model_channels)
        self.adaLN_modulation = AdaLNSingle(c.model_channels)

        self.blocks = [
            ModulatedDiTCrossBlock(
                channels=c.model_channels,
                ctx_channels=c.cond_channels,
                num_heads=c.num_heads,
                mlp_ratio=c.mlp_ratio,
            )
            for _ in range(c.num_blocks)
        ]

        self.out_layer = nn.Linear(c.model_channels, c.out_channels, bias=True)

    def __call__(
        self,
        x: mx.array,
        coords: mx.array,
        t: mx.array,
        cond: mx.array,
        *,
        concat_cond: mx.array | None = None,
    ) -> mx.array:
        """Run one denoising step.

        Parameters
        ----------
        x : mx.array
            ``[L, in_channels]`` per-voxel noisy latent.
        coords : mx.array
            ``[L, 3]`` int latent-grid coordinates. Required for RoPE-3D
            inside the self-attention.
        t : mx.array
            ``[1]`` (or scalar reshaped to ``[1]``) timestep in
            ``[0, 1000]``. The pipeline passes ``1000 * t`` per
            ``flow_euler.py:45``.
        cond : mx.array
            ``[B, M, cond_channels]`` (or ``[M, cond_channels]``) DINOv3
            image features for cross-attention.
        concat_cond : mx.array or None
            Optional ``[L, in_channels_concat]`` features concatenated
            channel-wise to ``x`` before ``input_layer``. Used by the
            texture DiT to receive the shape latent.

        Returns
        -------
        mx.array
            ``[L, out_channels]`` predicted velocity in the rectified-flow
            sense.
        """
        if concat_cond is not None:
            x = mx.concatenate([x, concat_cond], axis=-1)
        h = self.input_layer(x)
        t_emb = self.adaLN_modulation(self.t_embedder(t))  # [1, 9216]
        for block in self.blocks:
            h = block(h, coords, t_emb, cond)
        h = mx.fast.layer_norm(h, weight=None, bias=None, eps=1e-5)
        return self.out_layer(h)


class SparseStructureFlowModel(nn.Module):
    """Stage 1 — sparse-structure DiT. **Dense** small grid (16³ × 8 ch).

    Stub for now; the dense variant uses the same block class but runs
    over a small full-grid tensor with precomputed RoPE phases. Lands once
    we wire stage 1 into the pipeline (the bigger gain is from stages 2/3
    which dominate inference time at the 1024³ output).
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("SparseStructureFlowModel lands with the stage-1 pipeline wiring")


__all__ = ["SLatFlowConfig", "SLatFlowModel", "SparseStructureFlowModel"]
