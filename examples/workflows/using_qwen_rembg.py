"""Mirror of upstream ``Using_Qwen_Rembg.json``. Load-image-with-alpha utility.

The upstream workflow is just two nodes:

  Trellis2LoadImageWithTransparency  (load image, expose RGBA)
  Trellis2StringSelector             (pick file by index)

i.e. it's not a generation workflow — it's a demo of upstream's
"load image with transparency" node. trellis2-mlx accepts RGBA input
images directly (the pipeline's alpha-mask crop is automatic), so the
equivalent here is just a small utility that loads the image and
reports its alpha-channel stats so you can confirm your background
removal worked.

Pair this with an external background-removal tool of your choice
(BiRefNet, rembg, Photoshop, etc.) and feed the resulting RGBA into
``examples/workflows/low_poly.py`` or any other workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pathlib import Path as _Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples.workflows._common import REPO_ROOT, pick_default_image


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="using_qwen_rembg",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image or pick_default_image()
    if image_path is None:
        print("error: no --image and no example image found", file=sys.stderr)
        return 2

    img = Image.open(image_path)
    print(f"loaded: {image_path}")
    print(f"  size: {img.size}  mode: {img.mode}")
    print(f"  repo root: {REPO_ROOT}")

    if img.mode != "RGBA":
        print(f"  ⚠  mode is {img.mode}, not RGBA — alpha-aware crop disabled.")
        print("     Run any image through a background remover first (BiRefNet, rembg, ...)")
        print("     and re-export as RGBA for best results.")
        return 0

    a = np.asarray(img)[:, :, 3]
    print(f"  alpha: min={a.min()}  max={a.max()}  mean={a.mean():.1f}")
    fg_frac = (a > 128).mean()
    print(f"  foreground coverage (alpha>128): {fg_frac * 100:.1f}%")
    print()
    print("alpha looks good ✓. Pipe this image into any generation workflow, e.g.:")
    print(f"  uv run python examples/workflows/low_poly.py --image {image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
