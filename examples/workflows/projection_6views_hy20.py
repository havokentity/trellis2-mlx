"""Mirror of upstream ``Projection_6Views_Hy20.json``.

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
        prog='projection_6views_hy20', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'Projection_6Views_Hy20.glb')
    parser.add_argument('--mesh', type=Path, default=None, help='External mesh')
    parser.add_argument('--views', type=int, default=6, help='Number of projection views')
    return not_implemented_stub(
        parser, workflow_json='Projection_6Views_Hy20.json',
        missing_features=['Hunyuan3D 2.0 integration (external image-to-3D model)', 'MultiViewTexturing pipeline (project 6 views onto existing mesh)'],
        upstream_nodes=['Trellis2MeshTexturing', 'Trellis2MultiViewTexturing'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
