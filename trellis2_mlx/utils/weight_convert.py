"""HF checkpoint → MLX state dict converter.

Implements ``PHASE0_SPEC.md §7``. The TRELLIS.2 stack is heterogeneous —
DINOv3-L lives in `transformers`, the DiTs and VAEs live in
`microsoft/TRELLIS.2-4B` safetensors. This module exposes one converter per
sub-model; the master inventory is in ``docs/weight-inventory.{md,json}``.

Currently implemented:

* :func:`dinov3_from_pt_state_dict` — given a ``transformers`` DINOv3ViTModel
  PyTorch ``state_dict`` (numpy arrays), produce an MLX-style flat mapping
  that :class:`trellis2_mlx.models.dinov3.DINOv3L` accepts via
  ``mlx.utils.tree_unflatten``.

DiT and SC-VAE converters land alongside their model ports.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def _to_mx(value: Any) -> mx.array:
    """Wrap a numpy / safetensors tensor as an ``mx.array`` (NHWC-aware caller).

    The conversion is dtype-preserving for fp16/bf16/fp32. The caller is
    responsible for any layout transposes (e.g. NCHW→NHWC for conv weights).
    """
    return mx.array(value)


def dinov3_from_pt_state_dict(
    pt_state_dict: Mapping[str, Any],
    *,
    num_hidden_layers: int = 24,
) -> list[tuple[str, mx.array]]:
    """Convert a PyTorch DINOv3ViTModel state dict to MLX parameter pairs.

    The HF DINOv3ViTModel parameter names are mapped to the structure of
    :class:`trellis2_mlx.models.dinov3.DINOv3L`. Notable conversions:

    * ``embeddings.patch_embeddings.weight`` is a Conv2d weight in NCHW
      format ``[out, in, kH, kW]``; we transpose to MLX's NHWC layout
      ``[out, kH, kW, in]``.
    * Everything else is a Linear or scalar parameter and copies directly.

    Parameters
    ----------
    pt_state_dict : mapping
        ``{name: numpy_array}`` from ``state_dict()`` with ``.cpu().numpy()``
        applied to each tensor.
    num_hidden_layers : int
        Number of transformer layers. The HF state dict prefixes layers
        with ``model.layer.{i}.`` — we drop the ``model.`` and rename
        ``layer.{i}.`` to ``layers.{i}.``.

    Returns
    -------
    list[tuple[str, mx.array]]
        Suitable for ``DINOv3L.load_weights(...)``.
    """
    out: dict[str, mx.array] = {}

    # Patch embedding + token parameters
    pe_w = pt_state_dict["embeddings.patch_embeddings.weight"]
    # PT Conv2d: [out, in, kH, kW] → MLX Conv2d: [out, kH, kW, in]
    pe_w_mlx = _to_mx(pe_w).transpose(0, 2, 3, 1)
    out["embeddings.patch_embeddings.weight"] = pe_w_mlx
    if "embeddings.patch_embeddings.bias" in pt_state_dict:
        out["embeddings.patch_embeddings.bias"] = _to_mx(
            pt_state_dict["embeddings.patch_embeddings.bias"]
        )
    out["embeddings.cls_token"] = _to_mx(pt_state_dict["embeddings.cls_token"])
    out["embeddings.register_tokens"] = _to_mx(pt_state_dict["embeddings.register_tokens"])
    out["embeddings.mask_token"] = _to_mx(pt_state_dict["embeddings.mask_token"])

    # Transformer layers — HF prefix is "model.layer.{i}." in DINOv3ViTModel;
    # accept both "model.layer.{i}." and bare "layer.{i}." for flexibility.
    def layer_key_in(state: Mapping[str, Any], i: int, suffix: str) -> str:
        for prefix in (f"model.layer.{i}.", f"layer.{i}."):
            if f"{prefix}{suffix}" in state:
                return f"{prefix}{suffix}"
        raise KeyError(f"missing layer {i} parameter {suffix}")

    for i in range(num_hidden_layers):
        pre = f"layers.{i}."
        for k_suffix, dst_name in [
            ("norm1.weight", "norm1.weight"),
            ("norm1.bias", "norm1.bias"),
            ("attention.q_proj.weight", "attention.q_proj.weight"),
            ("attention.q_proj.bias", "attention.q_proj.bias"),
            ("attention.k_proj.weight", "attention.k_proj.weight"),
            ("attention.v_proj.weight", "attention.v_proj.weight"),
            ("attention.v_proj.bias", "attention.v_proj.bias"),
            ("attention.o_proj.weight", "attention.o_proj.weight"),
            ("attention.o_proj.bias", "attention.o_proj.bias"),
            ("layer_scale1.lambda1", "layer_scale1.lambda1"),
            ("norm2.weight", "norm2.weight"),
            ("norm2.bias", "norm2.bias"),
            ("mlp.up_proj.weight", "mlp.up_proj.weight"),
            ("mlp.up_proj.bias", "mlp.up_proj.bias"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight"),
            ("mlp.down_proj.bias", "mlp.down_proj.bias"),
            ("layer_scale2.lambda1", "layer_scale2.lambda1"),
        ]:
            src = layer_key_in(pt_state_dict, i, k_suffix)
            out[pre + dst_name] = _to_mx(pt_state_dict[src])

    # Final (learned) LayerNorm — not used by our forward, but stored so the
    # state dict round-trips and `load_weights` doesn't error on missing
    # parameters. transformers stores this at top-level `norm.{weight,bias}`.
    for k in ("norm.weight", "norm.bias"):
        if k in pt_state_dict:
            out[k] = _to_mx(pt_state_dict[k])

    return list(out.items())


def _convert_sparse_conv3d_weight(pt_w: Any) -> mx.array:
    """Permute a SparseConv3d weight from ``[Co, Kd, Kh, Kw, Ci]`` (upstream's
    flex_gemm layout) to ``[27, Ci, Co]`` (our SubMConv3 layout).

    The 27 axis is z-y-x scan order — slot ``kd*9 + kh*3 + kw`` corresponds
    to the offset ``(kd-1, kh-1, kw-1)``, matching
    :func:`trellis2_mlx.ovoxel.data.neighbor_offset_index`.

    See ``docs/weight-inventory.md`` (sparse-conv weight-layout note) and
    ``reference/microsoft-trellis2/trellis2/modules/sparse/conv/conv_flex_gemm.py:36``.
    """
    arr = np.asarray(pt_w)
    if arr.ndim != 5 or arr.shape[1:4] != (3, 3, 3):
        raise ValueError(f"expected SparseConv3d weight [Co, 3, 3, 3, Ci], got {arr.shape}")
    co, _, _, _, ci = arr.shape
    # [Co, Kd, Kh, Kw, Ci] → [Kd, Kh, Kw, Ci, Co] → [27, Ci, Co]
    return _to_mx(arr.transpose(1, 2, 3, 4, 0).reshape(27, ci, co))


def convnext_block_from_pt_state_dict(
    pt_state_dict: Mapping[str, Any],
) -> list[tuple[str, mx.array]]:
    """Convert one SC-VAE ConvNeXt block's PT state dict to MLX param pairs.

    Expected input keys (per the upstream ``SparseConvNeXtBlock3d``):

    * ``conv.weight``  ``[Co, 3, 3, 3, Ci]`` (with ``Co == Ci`` in this block)
    * ``conv.bias``    ``[Co]``  (may be absent)
    * ``norm.weight``  ``[C]``
    * ``norm.bias``    ``[C]``
    * ``mlp.0.weight`` ``[mlp_C, C]``  (PT calls it ``mlp.0`` because it's
      the first Sequential entry)
    * ``mlp.0.bias``   ``[mlp_C]``
    * ``mlp.2.weight`` ``[C, mlp_C]``
    * ``mlp.2.bias``   ``[C]``

    Output is suitable for ``SparseConvNeXtBlock3d.load_weights(...)``.
    """
    out: dict[str, mx.array] = {}
    out["conv_weight"] = _convert_sparse_conv3d_weight(pt_state_dict["conv.weight"])
    if "conv.bias" in pt_state_dict:
        out["conv_bias"] = _to_mx(pt_state_dict["conv.bias"])
    out["norm.weight"] = _to_mx(pt_state_dict["norm.weight"])
    out["norm.bias"] = _to_mx(pt_state_dict["norm.bias"])
    out["mlp_up.weight"] = _to_mx(pt_state_dict["mlp.0.weight"])
    out["mlp_up.bias"] = _to_mx(pt_state_dict["mlp.0.bias"])
    out["mlp_down.weight"] = _to_mx(pt_state_dict["mlp.2.weight"])
    out["mlp_down.bias"] = _to_mx(pt_state_dict["mlp.2.bias"])
    return list(out.items())


def c2s_upsample_block_from_pt_state_dict(
    pt_state_dict: Mapping[str, Any],
) -> list[tuple[str, mx.array]]:
    """Convert one ``SparseResBlockC2S3d`` PT state dict to MLX param pairs.

    Expected upstream keys:

    * ``to_subdiv.weight`` ``[8, in_C]``,    ``to_subdiv.bias`` ``[8]``
    * ``norm1.weight`` / ``norm1.bias``      (affine, ``[in_C]``)
    * ``conv1.weight``     ``[out_C * 8, 3, 3, 3, in_C]``
    * ``conv1.bias``       ``[out_C * 8]``
    * ``conv2.weight``     ``[out_C, 3, 3, 3, out_C]``
    * ``conv2.bias``       ``[out_C]``

    The norm2 sub-block is non-affine and has no stored params.

    Output is suitable for ``SparseResBlockC2S3d.load_weights(...)``.
    """
    out: dict[str, mx.array] = {
        "to_subdiv.weight": _to_mx(pt_state_dict["to_subdiv.weight"]),
        "to_subdiv.bias": _to_mx(pt_state_dict["to_subdiv.bias"]),
        "norm1.weight": _to_mx(pt_state_dict["norm1.weight"]),
        "norm1.bias": _to_mx(pt_state_dict["norm1.bias"]),
        "conv1_weight": _convert_sparse_conv3d_weight(pt_state_dict["conv1.weight"]),
        "conv1_bias": _to_mx(pt_state_dict["conv1.bias"]),
        "conv2_weight": _convert_sparse_conv3d_weight(pt_state_dict["conv2.weight"]),
        "conv2_bias": _to_mx(pt_state_dict["conv2.bias"]),
    }
    return list(out.items())


def dit_block_from_pt_state_dict(
    pt_state_dict: Mapping[str, Any],
) -> list[tuple[str, mx.array]]:
    """Convert one DiT block's PT state dict to MLX param pairs.

    All keys map verbatim except the FFN ``nn.Sequential``: upstream stores
    it under ``mlp.mlp.{0,2}.{weight,bias}`` (outer ``SparseFeedForwardNet``
    wraps an inner ``nn.Sequential``); our :class:`~trellis2_mlx.nn.dit_block.ModulatedDiTCrossBlock`
    drops the outer wrapper, so MLX paths are ``mlp.layers.{0,2}.{weight,bias}``.

    Expected upstream keys (per ``blocks.{i}.*``):

    * ``modulation`` ``[6*C]``
    * ``norm2.{weight,bias}`` ``[C]``  (affine; norm1 / norm3 have no params)
    * ``self_attn.to_qkv.{weight,bias}`` ``[3*C, C]`` / ``[3*C]``
    * ``self_attn.to_out.{weight,bias}`` ``[C, C]`` / ``[C]``
    * ``self_attn.{q,k}_rms_norm.gamma`` ``[H, head_dim]``
    * ``cross_attn.to_q.{weight,bias}`` ``[C, C]`` / ``[C]``
    * ``cross_attn.to_kv.{weight,bias}`` ``[2*C, ctx_C]`` / ``[2*C]``
    * ``cross_attn.to_out.{weight,bias}``
    * ``cross_attn.{q,k}_rms_norm.gamma``
    * ``mlp.mlp.0.{weight,bias}`` ``[intermediate, C]`` / ``[intermediate]``
    * ``mlp.mlp.2.{weight,bias}`` ``[C, intermediate]`` / ``[C]``
    """
    out: list[tuple[str, mx.array]] = []
    rename_pairs = [
        ("mlp.mlp.0.weight", "mlp.layers.0.weight"),
        ("mlp.mlp.0.bias", "mlp.layers.0.bias"),
        ("mlp.mlp.2.weight", "mlp.layers.2.weight"),
        ("mlp.mlp.2.bias", "mlp.layers.2.bias"),
    ]
    direct_keys = [
        "modulation",
        "norm2.weight",
        "norm2.bias",
        "self_attn.to_qkv.weight",
        "self_attn.to_qkv.bias",
        "self_attn.to_out.weight",
        "self_attn.to_out.bias",
        "self_attn.q_rms_norm.gamma",
        "self_attn.k_rms_norm.gamma",
        "cross_attn.to_q.weight",
        "cross_attn.to_q.bias",
        "cross_attn.to_kv.weight",
        "cross_attn.to_kv.bias",
        "cross_attn.to_out.weight",
        "cross_attn.to_out.bias",
        "cross_attn.q_rms_norm.gamma",
        "cross_attn.k_rms_norm.gamma",
    ]
    for k in direct_keys:
        if k in pt_state_dict:
            out.append((k, _to_mx(pt_state_dict[k])))
    for pt_k, mlx_k in rename_pairs:
        if pt_k in pt_state_dict:
            out.append((mlx_k, _to_mx(pt_state_dict[pt_k])))
    return out


def slat_flow_model_from_pt_state_dict(
    pt_state_dict: Mapping[str, Any],
    *,
    num_blocks: int = 30,
) -> list[tuple[str, mx.array]]:
    """Convert a full SLAT DiT (``SLatFlowModel``) PT state dict to MLX pairs.

    Maps:

    * ``input_layer.{weight, bias}`` → same
    * ``out_layer.{weight, bias}``   → same
    * ``t_embedder.mlp.{0,2}.{weight,bias}`` → ``t_embedder.mlp.layers.{0,2}.*``
      (PT stores the Sequential under ``mlp`` with bare integer keys; MLX's
      ``nn.Sequential`` exposes children under ``.layers.`` — see
      ``test_dit_block.py`` for the same pattern at block scale.)
    * ``adaLN_modulation.1.{weight, bias}`` → ``adaLN_modulation.mlp.layers.1.*``
      (PT places ``nn.Sequential(SiLU, Linear)`` directly at
      ``adaLN_modulation``; our :class:`~trellis2_mlx.nn.adaln.AdaLNSingle`
      wraps it under ``.mlp``.)
    * ``blocks.{i}.*``  → same, via :func:`dit_block_from_pt_state_dict`.

    Parameters
    ----------
    pt_state_dict : mapping
        ``{name: numpy_array}`` covering every parameter in the SLAT
        ``.safetensors``.
    num_blocks : int
        Block count (30 for every published checkpoint).
    """
    out: list[tuple[str, mx.array]] = []
    out.append(("input_layer.weight", _to_mx(pt_state_dict["input_layer.weight"])))
    out.append(("input_layer.bias", _to_mx(pt_state_dict["input_layer.bias"])))
    out.append(("out_layer.weight", _to_mx(pt_state_dict["out_layer.weight"])))
    out.append(("out_layer.bias", _to_mx(pt_state_dict["out_layer.bias"])))

    # TimestepEmbedder: PT Sequential at .mlp with bare integers; MLX needs
    # `.mlp.layers.{i}`.
    for i in (0, 2):
        w = pt_state_dict.get(f"t_embedder.mlp.{i}.weight")
        b = pt_state_dict.get(f"t_embedder.mlp.{i}.bias")
        if w is None or b is None:
            raise KeyError(f"missing t_embedder.mlp.{i}.* in state dict")
        out.append((f"t_embedder.mlp.layers.{i}.weight", _to_mx(w)))
        out.append((f"t_embedder.mlp.layers.{i}.bias", _to_mx(b)))

    # AdaLNSingle: PT puts nn.Sequential(SiLU, Linear) directly at
    # adaLN_modulation; our class wraps it under .mlp.
    out.append(
        (
            "adaLN_modulation.mlp.layers.1.weight",
            _to_mx(pt_state_dict["adaLN_modulation.1.weight"]),
        )
    )
    out.append(
        (
            "adaLN_modulation.mlp.layers.1.bias",
            _to_mx(pt_state_dict["adaLN_modulation.1.bias"]),
        )
    )

    # Blocks — delegate to per-block converter.
    for i in range(num_blocks):
        prefix_pt = f"blocks.{i}."
        block_state = {
            k[len(prefix_pt) :]: v for k, v in pt_state_dict.items() if k.startswith(prefix_pt)
        }
        for sub_name, arr in dit_block_from_pt_state_dict(block_state):
            out.append((f"blocks.{i}.{sub_name}", arr))

    return out


def ss_flow_model_from_pt_state_dict(
    pt_state_dict: Mapping[str, Any],
    *,
    num_blocks: int = 30,
) -> list[tuple[str, mx.array]]:
    """Convert a Sparse-Structure DiT (stage 1) PT state dict to MLX pairs.

    The architecture is identical to the SLAT DiT under the hood — our
    :class:`~trellis2_mlx.models.dit.SparseStructureFlowModel` embeds a
    :class:`~trellis2_mlx.models.dit.SLatFlowModel` under ``self.inner``,
    so this is :func:`slat_flow_model_from_pt_state_dict` with every key
    prefixed by ``inner.``.
    """
    inner_pairs = slat_flow_model_from_pt_state_dict(pt_state_dict, num_blocks=num_blocks)
    return [(f"inner.{k}", v) for k, v in inner_pairs]


def shape_decoder_from_pt_state_dict(
    pt_state_dict: Mapping[str, Any],
    *,
    num_blocks: tuple[int, ...] = (4, 16, 8, 4, 0),
) -> list[tuple[str, mx.array]]:
    """Convert the full SC-VAE shape decoder PT state dict to MLX param pairs.

    The upstream stores parameters under ``blocks.{stage}.{block}.{...}`` with
    a mixture of ConvNeXt blocks (positions ``0..num_blocks[stage]-1``) and
    one upsample block (position ``num_blocks[stage]``) per non-final stage.
    Plus ``from_latent.{weight, bias}`` and ``output_layer.{weight, bias}``
    at the top level.

    Parameters
    ----------
    pt_state_dict : mapping
        ``{name: numpy_array}`` covering every parameter in
        ``shape_dec_next_dc_f16c32_fp16.safetensors``.
    num_blocks : tuple of int
        Per-stage ConvNeXt block counts. Default matches the published
        ``shape_vae_next_dc_f16c32_fp16.json``: ``(4, 16, 8, 4, 0)``. The
        upsample slot at the end of each non-final stage is appended
        implicitly.

    Returns
    -------
    list[tuple[str, mx.array]]
        Suitable for ``ShapeDecoder.load_weights(...)``.
    """
    out: list[tuple[str, mx.array]] = []

    # from_latent and output_layer — plain Linear weights.
    out.append(("from_latent.weight", _to_mx(pt_state_dict["from_latent.weight"])))
    out.append(("from_latent.bias", _to_mx(pt_state_dict["from_latent.bias"])))
    out.append(("output_layer.weight", _to_mx(pt_state_dict["output_layer.weight"])))
    out.append(("output_layer.bias", _to_mx(pt_state_dict["output_layer.bias"])))

    # Per-stage blocks.
    for stage_idx, n_convnext in enumerate(num_blocks):
        # ConvNeXt blocks at positions 0..n_convnext-1
        for block_idx in range(n_convnext):
            prefix_pt = f"blocks.{stage_idx}.{block_idx}."
            prefix_mlx = f"blocks.{stage_idx}.{block_idx}."
            block_state = {
                k[len(prefix_pt) :]: v for k, v in pt_state_dict.items() if k.startswith(prefix_pt)
            }
            for sub_name, arr in convnext_block_from_pt_state_dict(block_state):
                out.append((prefix_mlx + sub_name, arr))
        # Upsample at the end of each non-final stage.
        if stage_idx < len(num_blocks) - 1:
            up_idx = n_convnext
            prefix_pt = f"blocks.{stage_idx}.{up_idx}."
            prefix_mlx = f"blocks.{stage_idx}.{up_idx}."
            block_state = {
                k[len(prefix_pt) :]: v for k, v in pt_state_dict.items() if k.startswith(prefix_pt)
            }
            for sub_name, arr in c2s_upsample_block_from_pt_state_dict(block_state):
                out.append((prefix_mlx + sub_name, arr))

    return out


def convert_checkpoint(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Convert the full TRELLIS.2-4B safetensors at ``src_dir`` to MLX format.

    Stub for the multi-model pipeline (DiTs + SC-VAEs); lands with the
    respective model ports.
    """
    raise NotImplementedError("multi-model conversion lands with the DiT/VAE ports")
