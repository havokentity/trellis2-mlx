"""DINOv3-L MLX port — parity tests against the ``transformers`` PyTorch reference.

Two modes:

* :func:`test_dinov3_tiny_random_init_parity` instantiates a *tiny* DINOv3
  config (4 layers, hidden 64, 4 heads), randomly initializes a PyTorch
  ``DINOv3ViTModel``, converts its state dict to MLX, runs both on a fixed
  random image, and asserts the outputs match to ``atol=1e-4``. This runs
  in seconds and does NOT require network or HF auth — it validates
  architecture and the weight converter.

* :func:`test_dinov3_l_full_config_random_init_parity` repeats with the real
  ViT-L/16 config (24 layers, hidden 1024, 16 heads) at the smallest useful
  image size (32×32 → 2×2 patch grid). Slower, but exercises the exact dims
  the production pipeline will use.

A third test loading the real ``facebook/dinov3-vitl16-pretrain-lvd1689m``
weights lives in ``test_dinov3_hf_weights.py`` and is skipped when there is
no HF token. The pipeline ``image_feature_extractor`` uses ``F.layer_norm``
without learned params on the final hidden state (see
``docs/open-questions-resolved.md`` Q3) — we reproduce that here for the
reference side.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812 — standard PyTorch alias

from trellis2_mlx.models.dinov3 import DINOv3L, DINOv3LConfig
from trellis2_mlx.utils.weight_convert import dinov3_from_pt_state_dict

pytestmark = pytest.mark.reference


def _pt_extract_features(pt_model: torch.nn.Module, image_nchw: torch.Tensor) -> torch.Tensor:
    """Mirror ``trellis2/modules/image_feature_extractor.py:81-92`` — embeddings →
    rope → encoder → parameter-free layer_norm. Bypasses ``pt_model.norm``.
    """
    hs = pt_model.embeddings(image_nchw, bool_masked_pos=None)
    pos = pt_model.rope_embeddings(image_nchw)
    for layer in pt_model.model.layer:
        hs = layer(hs, position_embeddings=pos)
    return F.layer_norm(hs, hs.shape[-1:])


def _make_pt_model(cfg: DINOv3LConfig) -> torch.nn.Module:
    """Build a PyTorch ``DINOv3ViTModel`` matching our MLX config."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    pt_cfg = DINOv3ViTConfig(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        num_register_tokens=cfg.num_register_tokens,
        rope_theta=cfg.rope_theta,
        layerscale_value=cfg.layerscale_value,
        layer_norm_eps=cfg.layer_norm_eps,
        use_gated_mlp=False,  # ViT-L uses plain MLP, not gated
        hidden_act="gelu",
        # DINOv3 quirk: K has no bias; Q/V/output do
        key_bias=False,
        query_bias=True,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    model = DINOv3ViTModel(pt_cfg).eval()
    return model


def _pt_state_dict_numpy(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {name: t.detach().cpu().numpy() for name, t in model.state_dict().items()}


def _max_abs_diff(a: mx.array, b: torch.Tensor) -> float:
    return float(np.abs(np.asarray(a) - b.detach().cpu().numpy()).max())


@pytest.fixture(autouse=True)
def _torch_seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


def _run_parity(cfg: DINOv3LConfig, image_size: int, atol: float) -> None:
    pt_model = _make_pt_model(cfg)
    state = _pt_state_dict_numpy(pt_model)

    # Same image for both backends
    img_nchw = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    pt_out = _pt_extract_features(pt_model, img_nchw)

    mlx_model = DINOv3L(cfg)
    mlx_model.load_weights(
        dinov3_from_pt_state_dict(state, num_hidden_layers=cfg.num_hidden_layers)
    )
    img_nhwc = mx.array(img_nchw.permute(0, 2, 3, 1).contiguous().numpy())
    mlx_out = mlx_model(img_nhwc)
    mx.eval(mlx_out)

    assert tuple(mlx_out.shape) == tuple(pt_out.shape), (
        f"shape mismatch: mlx {mlx_out.shape} vs pt {tuple(pt_out.shape)}"
    )
    diff = _max_abs_diff(mlx_out, pt_out)
    assert diff < atol, f"max |mlx - pt| = {diff:.3e} exceeds atol={atol:.3e}"


def test_dinov3_tiny_random_init_parity() -> None:
    """Tiny config — fast architectural check (~ <1 s)."""
    cfg = DINOv3LConfig(
        image_size=64,
        patch_size=16,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        num_register_tokens=4,
        rope_theta=100.0,
    )
    _run_parity(cfg, image_size=64, atol=1e-4)


@pytest.mark.slow
def test_dinov3_l_full_config_random_init_parity() -> None:
    """Production ViT-L/16 config at a tiny image — full-depth architectural check."""
    cfg = DINOv3LConfig()
    # 32×32 image → 2×2 patch grid — still ~1.3B-param-equivalent kernel sizes,
    # but tiny activation tensors keep the test under a few seconds on CPU.
    _run_parity(cfg, image_size=32, atol=2e-4)
