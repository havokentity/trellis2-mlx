"""Mirror of upstream ``Advanced_CustomSteps_MeshOnly.json``.

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
        prog='advanced_custom_steps_mesh_only', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'Advanced_CustomSteps_MeshOnly.glb')
    parser.add_argument('--ss-steps', type=int, default=12, help='SS-DiT steps')
    parser.add_argument('--shape-steps', type=int, default=12, help='Shape DiT steps')
    return not_implemented_stub(
        parser, workflow_json='Advanced_CustomSteps_MeshOnly.json',
        missing_features=['Decomposed generator nodes (ShapeGenerator, ShapeCascadeGenerator, SparseGenerator)'],
        upstream_nodes=['Trellis2ShapeGenerator', 'Trellis2ShapeCascadeGenerator', 'Trellis2SparseGenerator'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
