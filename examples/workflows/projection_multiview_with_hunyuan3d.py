"""Mirror of upstream ``Projection_MultiView_with_Hunyuan3D2.0.json``.

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
        prog='projection_multiview_with_hunyuan3d', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'Projection_MultiView_with_Hunyuan3D2.0.glb')
    parser.add_argument('--target-faces', type=int, default=500000, help='Decimation target')
    return not_implemented_stub(
        parser, workflow_json='Projection_MultiView_with_Hunyuan3D2.0.json',
        missing_features=['Hunyuan3D 2.0 mesh generation', 'Decomposed shape/cascade/sparse generator pipeline', 'MultiViewTexturing + PostProcess2 mesh cleanup'],
        upstream_nodes=['Trellis2ShapeGenerator', 'Trellis2ShapeCascadeGenerator', 'Trellis2SparseGenerator', 'Trellis2MeshTexturing', 'Trellis2MultiViewTexturing', 'Trellis2PostProcess2'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
