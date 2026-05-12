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


def convert_checkpoint(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Convert the full TRELLIS.2-4B safetensors at ``src_dir`` to MLX format.

    Stub for the multi-model pipeline (DiTs + SC-VAEs); lands with the
    respective model ports.
    """
    raise NotImplementedError("multi-model conversion lands with the DiT/VAE ports")
