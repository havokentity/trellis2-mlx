"""Mirror of upstream ``Watertight_Mesh.json``.

NOT YET IMPLEMENTED in trellis2-mlx.

This script exists so that every example workflow has a corresponding
Python entry point with the right CLI parameters; running it today
reports the upstream nodes that we have not yet ported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path  # noqa: F401  (Path is used by some --mesh / --video args)
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples.workflows._common import add_common_args, not_implemented_stub


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='watertight_mesh', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'Watertight_Mesh.glb')
    parser.add_argument('--target-faces', type=int, default=500000, help='Decimation target')
    parser.add_argument('--texture-resolution', type=int, default=2048, help='UV atlas size')
    return not_implemented_stub(
        parser, workflow_json='Watertight_Mesh.json',
        missing_features=['Decomposed shape pipeline (ShapeGenerator + ShapeCascadeGenerator)', 'TexSlatGenerator (decomposed texture stage)', 'VoxelToMesh closed-surface extractor (Trellis2VoxelToMesh)', 'PostProcess2 mesh cleanup'],
        upstream_nodes=['Trellis2ShapeGenerator', 'Trellis2ShapeCascadeGenerator', 'Trellis2SparseGenerator', 'Trellis2TexSlatGenerator', 'Trellis2VoxelToMesh', 'Trellis2PostProcess2'],
        notes='Closed-mesh extraction needs a different surface extractor than Flexible Dual Grid; not yet implemented.',
    )


if __name__ == "__main__":
    sys.exit(main())
