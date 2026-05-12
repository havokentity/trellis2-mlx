"""Modulated DiT cross-attention block.

Implements ``PHASE0_SPEC.md §4.4`` with the corrections from
``docs/open-questions-resolved.md``:

* Only **self-attn** and **FFN** are AdaLN-modulated; cross-attn keeps a
  plain learned-affine LayerNorm and no gating.
* ``share_mod = True`` is fixed for the published checkpoints: a shared
  ``SiLU + Linear(C, 6C)`` lives at the model root; each block carries a
  learned ``[6C]`` bias.
* QK-Norm on both self- and cross-attention.
* 3D RoPE on self-attention only (image-feature context is unordered).
* FFN is plain ``Linear → GELU(tanh approx) → Linear`` with
  ``mlp_ratio = 5.3334`` → intermediate dim 8192 for ``hidden = 1536``.

Mirrors ``reference/microsoft-trellis2/trellis2/modules/sparse/transformer/modulated.py:ModulatedSparseTransformerCrossBlock``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from trellis2_mlx.nn.sparse_attn import SparseCrossAttention, SparseSelfAttention


class ModulatedDiTCrossBlock(nn.Module):
    """One DiT transformer block — self-attn + cross-attn + FFN with
    AdaLN-single modulation on the self-attn and FFN branches.

    Block parameters (matches checkpoint key paths):

    * ``modulation`` ``[6 * channels]`` — per-block learned bias added to
      the shared modulation tensor before chunking.
    * ``norm2.weight``, ``norm2.bias`` ``[channels]`` — the *only*
      learned-affine LayerNorm in the block (between the self-attn add
      and the cross-attn). ``norm1`` and ``norm3`` are non-affine.
    * ``self_attn.{to_qkv,to_out,q_rms_norm,k_rms_norm}`` parameters.
    * ``cross_attn.{to_q,to_kv,to_out,q_rms_norm,k_rms_norm}`` parameters.
    * ``mlp.mlp.0.weight/bias`` ``[intermediate, channels]``,
      ``mlp.mlp.2.weight/bias`` ``[channels, intermediate]`` — note the
      ``mlp.mlp.*`` double prefix matches upstream ``nn.Sequential``.

    Parameters
    ----------
    channels : int
        Hidden dim (1536 for all three published DiTs).
    ctx_channels : int
        Image-feature dim (1024 for DINOv3-L).
    num_heads : int
        Number of attention heads (12 for all three published DiTs).
    mlp_ratio : float
        FFN expansion factor. Checkpoint uses ``5.3334`` (intermediate
        dim 8192 for ``hidden = 1536``).
    """

    def __init__(
        self,
        channels: int = 1536,
        ctx_channels: int = 1024,
        num_heads: int = 12,
        mlp_ratio: float = 5.3334,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.ctx_channels = ctx_channels
        self.num_heads = num_heads
        intermediate = int(channels * mlp_ratio)

        # Per-block learned bias on top of the shared modulation MLP output.
        # The 6 chunks (along the channel dim) are
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        self.modulation = mx.zeros((6 * channels,))

        # Three LayerNorms — only norm2 is affine in upstream.
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, affine=False)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, affine=True)
        self.norm3 = nn.LayerNorm(channels, eps=1e-6, affine=False)

        self.self_attn = SparseSelfAttention(channels, num_heads)
        self.cross_attn = SparseCrossAttention(channels, ctx_channels, num_heads)

        # FFN: Linear → GELU(tanh) → Linear. Matches upstream
        # SparseFeedForwardNet at trellis2/modules/sparse/transformer/blocks.py:11.
        # The double "mlp.mlp" prefix matches upstream nn.Sequential(...) layout.
        self.mlp = nn.Sequential(
            nn.Linear(channels, intermediate, bias=True),
            nn.GELU(approx="tanh"),
            nn.Linear(intermediate, channels, bias=True),
        )

    def __call__(
        self,
        x: mx.array,
        coords: mx.array,
        modulation: mx.array,
        context: mx.array,
    ) -> mx.array:
        """Apply the block.

        Parameters
        ----------
        x : mx.array
            ``[L, channels]`` per-voxel features.
        coords : mx.array
            ``[L, 3]`` voxel coordinates (for RoPE-3D in self-attention).
        modulation : mx.array
            ``[6 * channels]`` or ``[1, 6 * channels]`` shared modulation
            output. We add this block's learned ``self.modulation`` bias
            before chunking; B=1 broadcasts trivially against ``[L, C]``.
        context : mx.array
            ``[B, M, ctx_channels]`` or ``[M, ctx_channels]`` image
            features (DINOv3 patch + register + CLS tokens).
        """
        # Compute six modulation scalars from the shared + per-block sum.
        mod = modulation.reshape(-1) + self.modulation  # [6 * channels]
        chunks = mod.reshape(6, self.channels)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_mlp, scale_mlp, gate_mlp = chunks[3], chunks[4], chunks[5]

        # Self-attn branch — modulated.
        h = self.norm1(x)
        h = h * (1.0 + scale_msa) + shift_msa
        h = self.self_attn(h, coords)
        h = h * gate_msa
        x = x + h

        # Cross-attn branch — affine LN only, no modulation.
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h

        # FFN branch — modulated.
        h = self.norm3(x)
        h = h * (1.0 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x


__all__ = ["ModulatedDiTCrossBlock"]
