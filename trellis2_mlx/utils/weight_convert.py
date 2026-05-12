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


def convert_checkpoint(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Convert the full TRELLIS.2-4B safetensors at ``src_dir`` to MLX format.

    Stub for the multi-model pipeline (DiTs + SC-VAEs); lands with the
    respective model ports.
    """
    raise NotImplementedError("multi-model conversion lands with the DiT/VAE ports")
