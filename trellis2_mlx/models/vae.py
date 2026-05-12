"""SC-VAE shape decoder, MLX port.

Implements the FlexiDualGridVaeDecoder from
``reference/microsoft-trellis2/trellis2/models/sc_vaes/sparse_unet_vae.py:SparseUnetVaeDecoder``
plus the final ``(v, δ, γ)`` split from
``reference/microsoft-trellis2/trellis2/models/sc_vaes/fdg_vae.py:53-110``.

Shape per the published config (``configs/scvae/shape_vae_next_dc_f16c32_fp16.json``):

* ``from_latent``       — Linear(32 → 1024)
* Stage 0: 4 × ConvNeXt @ 1024 ch  → Upsample(1024 → 512)
* Stage 1: 16 × ConvNeXt @ 512 ch  → Upsample(512 → 256)
* Stage 2: 8 × ConvNeXt @ 256 ch   → Upsample(256 → 128)
* Stage 3: 4 × ConvNeXt @ 128 ch   → Upsample(128 → 64)
* Stage 4: 0 blocks (just receives the upsampled output)
* ``F.layer_norm`` (no learned params) over the final hidden state
* ``output_layer``      — Linear(64 → 7)
* Output split per ``fdg_vae.py:97-102``:
    * slots 0:3 → ``v = (1 + 2 ε) · sigmoid(·) - ε`` with ``ε = voxel_margin``
    * slots 3:6 → ``δ`` (raw logits; threshold at 0 for the dual-grid edges)
    * slots 6:7 → ``γ = softplus(·)``  (per-voxel quad-split weight)

Encoder + material decoder are deliberately left as stubs — the encoder is
needed only for fine-tuning support and lands then; the material decoder
shares the same architecture and gets a sibling class once the shape
decoder is end-to-end-validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from trellis2_mlx.nn.sparse_blocks import SparseConvNeXtBlock3d, SparseResBlockC2S3d
from trellis2_mlx.ovoxel.data import build_neighbor_table


@dataclass(frozen=True)
class ShapeDecoderConfig:
    """Static config for the SC-VAE shape decoder.

    Defaults match ``shape_vae_next_dc_f16c32_fp16.json``. ``voxel_margin`` is
    from ``fdg_vae.py:63`` (default 0.5 → dual vertex range ``[-0.5, 1.5]``).
    """

    latent_channels: int = 32
    out_channels: int = 7
    model_channels: tuple[int, ...] = (1024, 512, 256, 128, 64)
    num_blocks: tuple[int, ...] = (4, 16, 8, 4, 0)
    mlp_ratio: float = 4.0
    voxel_margin: float = 0.5


@dataclass
class ShapeDecoderOutput:
    """Decoded O-Voxel shape fields. See ``PHASE0_SPEC.md §3.1`` (corrected by
    ``docs/open-questions-resolved.md`` Q9).

    Attributes
    ----------
    coords : mx.array
        ``[L_fine, 3]`` int output-resolution voxel coordinates.
    v : mx.array
        ``[L_fine, 3]`` dual-vertex offsets in
        ``[-voxel_margin, 1 + voxel_margin]``.
    delta_logits : mx.array
        ``[L_fine, 3]`` raw edge-activity logits; threshold at 0 to recover
        the binary ``δ`` mask for mesh extraction.
    gamma : mx.array
        ``[L_fine, 1]`` per-voxel quad-split weight in ``(0, ∞)``.
    output_resolution : int
        Grid resolution at the output — ``coarse_resolution << len(model_channels)``
        for the standard 16× upsample (4 stages × 2).
    subdiv_logits : list[mx.array]
        Per-upsample-stage subdivision logits ``[L_coarse_i, 8]``. Useful for
        the training-time BCE supervision; ``None`` entries indicate stages
        whose subdivision was overridden by ``guide_subs``.
    """

    coords: mx.array
    v: mx.array
    delta_logits: mx.array
    gamma: mx.array
    output_resolution: int
    subdiv_logits: list[mx.array]


class ShapeDecoder(nn.Module):
    """SC-VAE shape decoder, MLX port.

    The decoder is stateless w.r.t. resolution — the caller supplies the
    initial coarse-grid coords and resolution and gets back fine-grid
    output. Four upsample stages double the resolution each time
    (4× factor 2 = 16× total). Initial coords typically come from the
    SLAT DiT at 32³ (for 512³ output) or 64³ (for 1024³ output).
    """

    def __init__(self, cfg: ShapeDecoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ShapeDecoderConfig()
        c = self.cfg

        # Linear projection from the SLAT-DiT latent to the first stage channels.
        self.from_latent = nn.Linear(c.latent_channels, c.model_channels[0], bias=True)

        # Per-stage block lists — match the upstream nn.ModuleList(nn.ModuleList(...))
        # layout so PT key paths transfer verbatim.
        self.blocks: list[list[nn.Module]] = []
        for stage_idx, n_convnext in enumerate(c.num_blocks):
            stage: list[nn.Module] = []
            for _ in range(n_convnext):
                stage.append(
                    SparseConvNeXtBlock3d(c.model_channels[stage_idx], mlp_ratio=c.mlp_ratio)
                )
            # All stages except the last get an upsample as their final block.
            if stage_idx < len(c.num_blocks) - 1:
                stage.append(
                    SparseResBlockC2S3d(
                        c.model_channels[stage_idx],
                        c.model_channels[stage_idx + 1],
                    )
                )
            self.blocks.append(stage)

        # Final per-voxel Linear → 7-channel head.
        self.output_layer = nn.Linear(c.model_channels[-1], c.out_channels, bias=True)

    def __call__(
        self,
        latent_feats: mx.array,
        coords: mx.array,
        coarse_resolution: int,
        *,
        guide_subs: list[mx.array | None] | None = None,
    ) -> ShapeDecoderOutput:
        """Decode a latent O-Voxel into shape outputs.

        Parameters
        ----------
        latent_feats : mx.array
            ``[L_coarse, latent_channels]`` per-active-voxel latents from the
            SLAT DiT.
        coords : mx.array
            ``[L_coarse, 3]`` int latent-grid coordinates.
        coarse_resolution : int
            Resolution of the latent grid (32 for 512³ output, 64 for 1024³).
        guide_subs : list of mx.array or None
            Optional ``[stages-1]`` list of pre-computed subdivision masks
            (one per upsample). When ``None`` (default), each upsample
            predicts its own from the features. Used by the material decoder
            in upstream to inherit the shape decoder's pruning structure.
        """
        if guide_subs is not None and len(guide_subs) != len(self.blocks) - 1:
            raise ValueError(
                f"guide_subs must have length {len(self.blocks) - 1}, got {len(guide_subs)}"
            )

        h = self.from_latent(latent_feats)
        nt = build_neighbor_table(coords, resolution=coarse_resolution)
        resolution = coarse_resolution
        subdiv_logits_per_stage: list[mx.array] = []

        for stage_idx, stage in enumerate(self.blocks):
            is_not_final_stage = stage_idx < len(self.blocks) - 1
            for block_idx, block in enumerate(stage):
                is_upsample = is_not_final_stage and block_idx == len(stage) - 1
                if is_upsample:
                    fine_resolution = resolution * 2
                    sub_override = guide_subs[stage_idx] if guide_subs is not None else None
                    h, coords, nt, subdiv_logits = block(
                        h,
                        coords,
                        nt,
                        fine_resolution=fine_resolution,
                        subdivision=sub_override,
                    )
                    subdiv_logits_per_stage.append(subdiv_logits)
                    resolution = fine_resolution
                else:
                    h = block(h, nt)

        # Final parameter-free LayerNorm + Linear head.
        h = mx.fast.layer_norm(h, weight=None, bias=None, eps=1e-5)
        h = self.output_layer(h)

        # Split the 7-channel head into (v, δ, γ) per fdg_vae.py:97-102.
        eps = self.cfg.voxel_margin
        v = (1.0 + 2.0 * eps) * mx.sigmoid(h[:, 0:3]) - eps
        delta_logits = h[:, 3:6]
        gamma = nn.softplus(h[:, 6:7])

        return ShapeDecoderOutput(
            coords=coords,
            v=v,
            delta_logits=delta_logits,
            gamma=gamma,
            output_resolution=resolution,
            subdiv_logits=subdiv_logits_per_stage,
        )


class SCVAEEncoder(nn.Module):
    """Sparse-conv encoder. Not needed for inference (we use Microsoft's
    pretrained checkpoint); included for fine-tuning support."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("SCVAEEncoder lands when fine-tuning support is wired up")


class SCVAEMaterialDecoder(nn.Module):
    """Material decoder producing per-active-voxel ``(c, m, r, α)``.

    Same architecture as :class:`ShapeDecoder` but with ``out_channels=6`` and
    ``guide_subs`` provided by the shape decoder so geometry and material
    share the same active set.

    Will be implemented once the shape decoder is end-to-end-validated; the
    block-level parity tests already cover the shared SubMConv3 / ConvNeXt /
    C2S internals.
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("SCVAEMaterialDecoder lands after shape-decoder smoke test")
