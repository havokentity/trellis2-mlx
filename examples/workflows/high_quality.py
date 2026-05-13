"""Mirror of upstream ``High_Quality.json``. 1024_cascade + refiner pass.

UPSTREAM WORKFLOW:

  Trellis2MeshWithVoxelAdvancedGenerator  mode=1024_cascade
  Trellis2ReconstructMeshWithQuad         mode=1, 1024 res
  Trellis2SimplifyMesh                    Cumesh
  Trellis2MeshRefiner                     image-conditioned refiner DiT
  Trellis2MeshTexturing                   defaults
  Trellis2SmoothNormals
  Trellis2FillHolesWithMeshlib

NOT YET IMPLEMENTED.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples.workflows._common import add_common_args, not_implemented_stub


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="high_quality", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, "HighQuality.glb")
    parser.add_argument("--mode", default="1024_cascade")
    parser.add_argument("--ss-steps", type=int, default=50)
    parser.add_argument("--slat-steps", type=int, default=50)
    parser.add_argument("--refiner-steps", type=int, default=25)
    parser.add_argument("--target-faces", type=int, default=1_000_000)
    return not_implemented_stub(
        parser, workflow_json="High_Quality.json",
        missing_features=[
            "1024_cascade generation mode",
            "MeshRefiner DiT",
            "ReconstructMeshWithQuad",
        ],
        upstream_nodes=[
            "Trellis2MeshWithVoxelAdvancedGenerator(mode=1024_cascade)",
            "Trellis2MeshRefiner",
            "Trellis2ReconstructMeshWithQuad",
            "Trellis2MeshTexturing",
        ],
        notes="Use examples/workflows/advanced.py for the closest 512-mode equivalent.",
    )


if __name__ == "__main__":
    sys.exit(main())
