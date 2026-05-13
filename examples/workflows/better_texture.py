"""Mirror of upstream ``Better_Texture.json``. Textured mesh with explicit texture knobs.

Upstream workflow:

  Trellis2MeshWithVoxelAdvancedGenerator  mode=1024  steps=12  cfg=6.5
  Trellis2Remesh                          1024 res
  Trellis2SimplifyMesh                    target_faces=500_000
  Trellis2FillHolesWithMeshlib            defaults
  Trellis2MeshTexturing                   tex_res=4096  alpha=OPAQUE

trellis2-mlx implementation:

* Mode=1024 falls back to 1024_cascade (same output resolution).
* Texture is vertex-color baking from the material decoder; the upstream
  4096² texture atlas needs UV-unwrap + rasterizer which are not in
  trellis2-mlx yet. At 500k faces the vertex-color approach is still
  meaningful but won't show fine detail like a 4K texture would.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples.workflows._common import (
    add_common_args,
    load_pipeline,
    resolve_image,
    resolve_pipeline_type,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="better_texture",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, "BetterTexture.glb")
    parser.add_argument("--mode", default="1024",
                        choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--target-faces", type=int, default=500_000)
    parser.add_argument("--texture-resolution", type=int, default=4096,
                        help="UV-atlas texture size (advisory — vertex-color path used today).")
    args = parser.parse_args()

    pipeline_type = resolve_pipeline_type(args.mode)
    if args.texture_resolution != 4096:
        print(f"  note: --texture-resolution={args.texture_resolution} is advisory "
              "(vertex-color path used today)")

    image, _ = resolve_image(args.image)
    pipeline = load_pipeline(args.seed, with_texture=True, pipeline_type=pipeline_type)

    print(f"running pipeline (mode={pipeline_type}) ...")
    t0 = time.perf_counter()
    result = pipeline.run(image)
    print(
        f"  generation: {time.perf_counter() - t0:.1f} s  →  "
        f"{result.vertices.shape[0]:,} verts, {result.faces.shape[0]:,} tris @ {result.output_resolution}³"
    )

    print(f"exporting GLB (fill_holes=True, target_faces={args.target_faces:,}) ...")
    t0 = time.perf_counter()
    written = pipeline.export_glb(
        result, args.output,
        repair=True, fill_holes=True, target_faces=args.target_faces, verbose=True,
    )
    print(f"  export+repair: {time.perf_counter() - t0:.1f} s")
    print(f"wrote GLB: {written}  ({written.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
