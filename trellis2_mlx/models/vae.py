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
                # Force materialization so MLX can free the previous block's
                # lazy-graph buffers. Without this the decoder accumulates the
                # entire 32-block forward in the graph at once — at realistic
                # active-set sizes (~30K coarse → up to millions fine) that
                # blows past Metal's 86 GB single-buffer cap.
                mx.eval(h, coords, nt)
                mx.clear_cache()

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


# ── SS-VAE decoder (dense; from microsoft/TRELLIS-image-large) ───────────


@dataclass(frozen=True)
class SparseStructureDecoderConfig:
    """Static config for the dense SS-VAE decoder.

    Matches ``microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16``:
    8-channel latent in, 1-channel occupancy out, 3 stages with channels
    ``[512, 128, 32]`` and 2 residual blocks each (plus 2 in the middle).
    Two upsample stages (×2 each) take 16³ → 64³.
    """

    out_channels: int = 1
    latent_channels: int = 8
    num_res_blocks: int = 2
    num_res_blocks_middle: int = 2
    channels: tuple[int, ...] = (512, 128, 32)


def _pixel_shuffle_3d_ndhwc(x: mx.array) -> mx.array:
    """3D pixel shuffle (factor=2) on an NDHWC tensor.

    Mirrors ``reference/microsoft-trellis2/trellis2/modules/spatial.py:pixel_shuffle_3d``
    EXACTLY: input channel index ``j`` is laid out as
    ``j = 8 * c_ + 4 * kd + 2 * kh + kw``, with ``c_`` *slowest*. That means
    the reshape must split the ``C*8`` dim into ``[C, 2, 2, 2]`` (C slowest),
    NOT ``[2, 2, 2, C]`` (C fastest). The trained SS-VAE conv1 weights were
    learnt with upstream's slot ordering — any other ordering misroutes the
    channels to the wrong spatial positions and the decoder produces
    garbage (positives almost everywhere).

    ``[B, D, H, W, C*8] → [B, 2D, 2H, 2W, C]``.
    """
    b, d, h, w, c8 = x.shape
    if c8 % 8 != 0:
        raise ValueError(f"channels must be divisible by 8 for ×2 pixel-shuffle; got {c8}")
    c = c8 // 8
    # Reshape: C dim splits as [C, 2, 2, 2] — c_ slowest, kw fastest (matches
    # upstream's [C_, 2, 2, 2] split).
    x = x.reshape(b, d, h, w, c, 2, 2, 2)
    # Permute to interleave kernel dims with spatial dims and put C last:
    # current positions: [b, d, h, w, c, kd, kh, kw] = [0, 1, 2, 3, 4, 5, 6, 7]
    # target:            [b, d, kd, h, kh, w, kw, c] = [0, 1, 5, 2, 6, 3, 7, 4]
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(b, d * 2, h * 2, w * 2, c)


class _SSResBlock3d(nn.Module):
    """ResBlock3d from sparse_structure_vae.py:22-47 — dense Conv3d.

    norm1 + SiLU + conv1 + norm2 + SiLU + conv2 + skip. With
    ``in_channels == out_channels`` (the only case used in the published
    checkpoint), ``skip_connection`` is ``Identity`` (no learned params).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        # In MLX, nn.LayerNorm normalizes over the last axis. For NDHWC
        # tensors that's the channel axis — exactly what upstream's
        # ChannelLayerNorm32 (which moves channels to last, then LN) does.
        self.norm1 = nn.LayerNorm(channels, eps=1e-5, affine=True)
        self.norm2 = nn.LayerNorm(channels, eps=1e-5, affine=True)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        return h + x


class _SSUpsampleBlock3d(nn.Module):
    """UpsampleBlock3d (Conv3d expanding ×8 + pixel_shuffle_3d).

    Mirrors upstream's "conv" mode in sparse_structure_vae.py:75-98.
    ``Conv3d(in, out * 8, kernel=3, padding=1)`` followed by a ×2 spatial
    pixel-shuffle.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels * 8, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return _pixel_shuffle_3d_ndhwc(self.conv(x))


class SparseStructureDecoder(nn.Module):
    """SS-VAE decoder — dense conv-3d net from latent (16³ × 8ch) to a
    64³ × 1ch occupancy field.

    Implements
    ``reference/microsoft-trellis2/trellis2/models/sparse_structure_vae.py:SparseStructureDecoder``.
    The pipeline thresholds the output > 0 then max-pools to the chosen
    stage-1 resolution (32 or 64) and uses the resulting nonzero coords as
    the active voxel set for the SLAT DiTs.
    """

    def __init__(self, cfg: SparseStructureDecoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SparseStructureDecoderConfig()
        c = self.cfg

        self.input_layer = nn.Conv3d(c.latent_channels, c.channels[0], kernel_size=3, padding=1)

        # 2 middle res blocks at channels[0]
        self.middle_block = [_SSResBlock3d(c.channels[0]) for _ in range(c.num_res_blocks_middle)]

        # Per-stage: num_res_blocks at channels[i], then Upsample(i → i+1)
        # for i < len(channels) - 1.
        self.blocks: list[nn.Module] = []
        for i, ch in enumerate(c.channels):
            for _ in range(c.num_res_blocks):
                self.blocks.append(_SSResBlock3d(ch))
            if i < len(c.channels) - 1:
                self.blocks.append(_SSUpsampleBlock3d(ch, c.channels[i + 1]))

        # Out: norm + SiLU + Conv3d(channels[-1] → out_channels). Upstream stores
        # this as nn.Sequential(norm, silu, conv) — keys are out_layer.0.* (norm)
        # and out_layer.2.* (conv).
        self.out_layer = nn.Sequential(
            nn.LayerNorm(c.channels[-1], eps=1e-5, affine=True),
            nn.SiLU(),
            nn.Conv3d(c.channels[-1], c.out_channels, kernel_size=3, padding=1),
        )

    def __call__(self, z: mx.array) -> mx.array:
        """Decode a latent.

        Parameters
        ----------
        z : mx.array
            ``[B, latent_channels, D, H, W]`` (PT-style NCDHW) or
            ``[B, D, H, W, latent_channels]`` (NDHWC). The forward auto-
            detects which by checking which dim equals ``latent_channels``;
            outputs match the input convention.

        Returns
        -------
        mx.array
            Occupancy logits at 4× spatial resolution
            (``D * 4, H * 4, W * 4``). For 16³ input → 64³ output.
        """
        if z.ndim != 5:
            raise ValueError(f"z must be 5D; got {z.shape}")
        # Auto-detect NCDHW vs NDHWC.
        nchw_input = z.shape[1] == self.cfg.latent_channels
        if nchw_input:
            z = z.transpose(0, 2, 3, 4, 1)  # → NDHWC

        h = self.input_layer(z)
        for blk in self.middle_block:
            h = blk(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_layer(h)

        if nchw_input:
            h = h.transpose(0, 4, 1, 2, 3)  # back to NCDHW
        return h
