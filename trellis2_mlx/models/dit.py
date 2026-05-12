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


@dataclass(frozen=True)
class SparseStructureFlowConfig:
    """Static config for the stage-1 sparse-structure DiT.

    Defaults match ``configs/gen/ss_flow_img_dit_1_3B_64_bf16.json``:
    dense 16³ grid with 8 latent channels per voxel. The "_64_" in the
    checkpoint name refers to the SS-VAE's *output* resolution (16 × 4 = 64);
    the DiT itself runs at 16³.
    """

    resolution: int = 16
    in_channels: int = 8
    out_channels: int = 8
    model_channels: int = 1536
    cond_channels: int = 1024
    num_blocks: int = 30
    num_heads: int = 12
    mlp_ratio: float = 5.3334
    rope_base: float = 10000.0


class SparseStructureFlowModel(nn.Module):
    """Stage 1 — sparse-structure DiT (dense 16³ × 8ch).

    Reuses :class:`SLatFlowModel`'s blocks but accepts a **dense**
    ``[B, C, D, H, W]`` input (matching upstream
    ``sparse_structure_flow.py:224-247``). Internally:

    1. Reshape to ``[L, C_in]`` where ``L = D * H * W`` (all voxels of
       the dense grid are tokens).
    2. Build the static ``[L, 3]`` coords meshgrid once (cached per
       resolution).
    3. Run the same DiT stack.
    4. Reshape back to ``[B, C_out, D, H, W]``.

    For ``B = 1`` (inference) we drop the batch dim entirely inside the
    transformer. CFG is handled at the sampler level by calling the
    model twice with different conditioning.
    """

    def __init__(self, cfg: SparseStructureFlowConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SparseStructureFlowConfig()
        c = self.cfg

        # Reuse the SLatFlowModel for everything except the dense reshape.
        # Same blocks, same parameter paths — the weight converter is
        # bitwise-identical (slat_flow_model_from_pt_state_dict).
        slat_cfg = SLatFlowConfig(
            resolution=c.resolution,
            in_channels=c.in_channels,
            out_channels=c.out_channels,
            model_channels=c.model_channels,
            cond_channels=c.cond_channels,
            num_blocks=c.num_blocks,
            num_heads=c.num_heads,
            mlp_ratio=c.mlp_ratio,
            rope_base=c.rope_base,
        )
        # Embed the SLAT model under `inner` — the weight converter routes
        # every PT key through `inner.*` so the parameter paths line up.
        self.inner = SLatFlowModel(slat_cfg)

        # Precompute the dense coord meshgrid for RoPE-3D. Shape [L, 3]
        # with coord order (z, y, x) matching the rest of the codebase.
        r = c.resolution
        grids = mx.meshgrid(
            mx.arange(r, dtype=mx.int32),
            mx.arange(r, dtype=mx.int32),
            mx.arange(r, dtype=mx.int32),
            indexing="ij",
        )
        coords = mx.stack(list(grids), axis=-1).reshape(-1, 3)
        # Store as a frozen attribute (not a parameter — RoPE coords are static).
        self._coords = coords

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        """Run one denoising step on the dense small grid.

        Parameters
        ----------
        x : mx.array
            ``[B, C_in, D, H, W]`` or ``[C_in, D, H, W]`` dense latent.
            ``D = H = W = resolution`` is enforced.
        t : mx.array
            ``[B]`` (or ``[1]``) timestep in ``[0, 1000]``.
        cond : mx.array
            ``[B, M, cond_channels]`` or ``[M, cond_channels]`` DINOv3
            features.

        Returns
        -------
        mx.array
            Same shape and dtype as ``x``.
        """
        if x.ndim == 4:
            x = x[None]
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, D, H, W] or [C, D, H, W]; got {x.shape}")
        b, c_in, d, h, w = x.shape
        if d != h or h != w or d != self.cfg.resolution:
            raise ValueError(f"x spatial dims must be {self.cfg.resolution}³; got ({d}, {h}, {w})")
        if b != 1:
            raise NotImplementedError("SparseStructureFlowModel currently runs at B=1")

        # [B, C, D, H, W] → [L, C]  (with B=1)
        x_flat = x[0].reshape(c_in, -1).transpose(1, 0)  # [L, C]

        # Run the same SLAT-style forward. The SLAT model's __call__ takes
        # (x, coords, t, cond, *, concat_cond=None) — we have no concat_cond.
        out_flat = self.inner(x_flat, self._coords, t, cond)

        # Back to [B, C, D, H, W]
        out = out_flat.transpose(1, 0).reshape(self.cfg.out_channels, d, h, w)[None]
        return out


__all__ = [
    "SLatFlowConfig",
    "SLatFlowModel",
    "SparseStructureFlowConfig",
    "SparseStructureFlowModel",
]
