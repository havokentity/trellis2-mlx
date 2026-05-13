"""Mirror of upstream ``Only_Mesh_Simple.json``. Geometry-only export.

Upstream workflow:

  Trellis2MeshWithVoxelGenerator   seed=*  mode=1024_cascade  steps=12 ...
  Trellis2Remesh                   1024 res, smooth=True, decimate=True
  Trellis2SimplifyMesh             target_faces=2_000_000
  Trellis2FillHolesWithMeshlib     defaults
  → 90° rotated trimesh → GLB

trellis2-mlx implementation:

* Generator runs at mode=512 (1024_cascade not yet implemented).
* No texture / vertex-color baking — geometry only, much faster.
* Default ``--target-faces 2_000_000`` matches upstream; on our 512
  path the raw mesh is around 1-2M so this often skips decimation.
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
        prog="only_mesh_simple",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, "OnlyMeshSimple.glb")
    parser.add_argument(
        "--mode",
        default="1024_cascade",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        help="Generator mode (upstream default: 1024_cascade).",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=2_000_000,
        help="Quadric decimation target (upstream default: 2_000_000).",
    )
    args = parser.parse_args()

    pipeline_type = resolve_pipeline_type(args.mode)
    image, _ = resolve_image(args.image)
    pipeline = load_pipeline(args.seed, with_texture=False, pipeline_type=pipeline_type)

    print(f"running pipeline (mode={pipeline_type}) ...")
    t0 = time.perf_counter()
    result = pipeline.run(image)
    print(
        f"  generation: {time.perf_counter() - t0:.1f} s  →  "
        f"{result.vertices.shape[0]:,} verts, {result.faces.shape[0]:,} tris @ {result.output_resolution}³"
    )

    print(f"exporting GLB (fill_holes=True, target_faces={args.target_faces:,}) ...")
    t0 = time.perf_counter()
    # If target > raw face count, the decimator is a no-op (which is fine).
    written = pipeline.export_glb(
        result,
        args.output,
        repair=True,
        fill_holes=True,
        target_faces=args.target_faces,
        verbose=True,
    )
    print(f"  export+repair: {time.perf_counter() - t0:.1f} s")
    print(f"wrote GLB: {written}  ({written.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
