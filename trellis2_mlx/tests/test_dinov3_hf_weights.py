"""End-to-end DINOv3-L parity test using the real Meta pretrained weights.

Skipped unless ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) is set and the
account has accepted Meta's gated-access terms for
``facebook/dinov3-vitl16-pretrain-lvd1689m`` (see
<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>).

What it does:

1. Downloads the HF checkpoint (cached after first run).
2. Runs the PyTorch model with the same ``extract_features`` recipe that
   ``trellis2/modules/image_feature_extractor.py`` uses.
3. Runs our MLX :class:`trellis2_mlx.models.dinov3.DINOv3L` on the same
   image.
4. Asserts the outputs match to ``atol=5e-3`` (fp32 path on a 24-layer ViT
   accumulates a few ulps).
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812 — standard PyTorch alias
from PIL import Image

from trellis2_mlx.models.dinov3 import DINOv3L, DINOv3LConfig
from trellis2_mlx.utils.weight_convert import dinov3_from_pt_state_dict

pytestmark = [pytest.mark.reference, pytest.mark.slow]

_HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _have_hf_token() -> bool:
    return (
        any(os.environ.get(v) for v in _HF_TOKEN_VARS)
        or Path(os.path.expanduser("~/.cache/huggingface/token")).exists()
    )


def _sample_image_path() -> Path:
    """Pick the first upstream example image if available."""
    candidates = sorted(Path("reference/microsoft-trellis2/assets/example_image").glob("*.webp"))
    if not candidates:
        pytest.skip("no upstream example image available")
    return candidates[0]


def _preprocess_pt(image: Image.Image, image_size: int) -> torch.Tensor:
    """Mirror ``trellis2/modules/image_feature_extractor.py:107-116``."""
    img = image.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


@pytest.mark.skipif(not _have_hf_token(), reason="HF_TOKEN not set / no cached token")
@pytest.mark.parametrize("image_size", [512, 1024])
def test_dinov3_l_hf_weights_parity(image_size: int) -> None:
    from transformers import DINOv3ViTModel

    image = Image.open(_sample_image_path())
    pt_image = _preprocess_pt(image, image_size)

    pt_model = DINOv3ViTModel.from_pretrained(_DINOV3_REPO).eval()

    with torch.no_grad():
        hs = pt_model.embeddings(pt_image, bool_masked_pos=None)
        pos = pt_model.rope_embeddings(pt_image)
        for layer in pt_model.model.layer:
            hs = layer(hs, position_embeddings=pos)
        pt_out = F.layer_norm(hs, hs.shape[-1:])

    cfg = DINOv3LConfig(image_size=image_size)
    mlx_model = DINOv3L(cfg)
    state = {k: v.detach().cpu().numpy() for k, v in pt_model.state_dict().items()}
    mlx_model.load_weights(
        dinov3_from_pt_state_dict(state, num_hidden_layers=cfg.num_hidden_layers)
    )

    mlx_image = mx.array(pt_image.permute(0, 2, 3, 1).contiguous().numpy())
    mlx_out = mlx_model(mlx_image)
    mx.eval(mlx_out)

    diff = float(np.abs(np.asarray(mlx_out) - pt_out.numpy()).max())
    assert diff < 5e-3, f"max |mlx - pt| = {diff:.3e} exceeds 5e-3 at image_size={image_size}"
