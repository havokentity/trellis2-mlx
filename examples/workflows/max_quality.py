"""Mirror of upstream ``Max_Quality.json``. Highest-quality pipeline (1536_cascade).

UPSTREAM WORKFLOW:

  Trellis2MeshWithVoxelAdvancedGenerator  mode=1536_cascade  ss_steps=50  cfg=6.5
  Trellis2ReconstructMeshWithQuad         mode=1, 1024 res
  Trellis2SimplifyMesh                    target_faces=2_000_000 Cumesh
  Trellis2FillHolesWithMeshlib            defaults
  Trellis2MeshRefiner                     refiner DiT (image-conditioned)
  Trellis2ReconstructMeshWithQuad         second pass
  Trellis2SimplifyMesh                    2_000_000 Cumesh
  Trellis2FillHolesWithMeshlib            defaults
  Trellis2MeshTexturing                   mesh_res=1536  tex_res=4096
  Trellis2SmoothNormals                   default

NOT YET IMPLEMENTED IN TRELLIS2-MLX.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples.workflows._common import add_common_args, not_implemented_stub


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="max_quality",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, "MaxQuality.glb")
    parser.add_argument("--mode", default="1536_cascade")
    parser.add_argument("--ss-steps", type=int, default=50)
    parser.add_argument("--slat-steps", type=int, default=50)
    parser.add_argument("--refiner-steps", type=int, default=25)
    parser.add_argument("--target-faces", type=int, default=2_000_000)
    parser.add_argument("--texture-resolution", type=int, default=4096)
    return not_implemented_stub(
        parser,
        workflow_json="Max_Quality.json",
        missing_features=[
            "1536_cascade generation mode (we only have 512 today)",
            "MeshRefiner DiT (second-pass image-guided geometry refinement)",
            "ReconstructMeshWithQuad (quad re-meshing pass at 1024³)",
            "UV unwrap + 4096² texture-atlas baking (we use per-vertex colors)",
        ],
        upstream_nodes=[
            "Trellis2MeshWithVoxelAdvancedGenerator(mode=1536_cascade)",
            "Trellis2ReconstructMeshWithQuad",
            "Trellis2MeshRefiner",
            "Trellis2MeshTexturing(tex_res=4096)",
        ],
        notes=(
            "For the closest currently-achievable result, use:\n"
            "  uv run python examples/workflows/low_poly.py --target-faces 2000000\n"
            "which runs the 512 path then decimates to 2M faces."
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
