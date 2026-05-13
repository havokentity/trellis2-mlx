"""Mirror of upstream ``MultiViews_TextureMesh.json``.

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
        prog='multi_views_texture_mesh', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'MultiViews_TextureMesh.glb')
    parser.add_argument('--mesh', type=Path, default=None, help='External mesh')
    parser.add_argument('--images', type=str, default=None, help='Comma-separated view images')
    return not_implemented_stub(
        parser, workflow_json='MultiViews_TextureMesh.json',
        missing_features=['MultiViewTexturing (project multiple views onto an existing mesh and texture-bake)', 'Trellis2LoadMesh'],
        upstream_nodes=['Trellis2LoadMesh', 'Trellis2MeshTexturingMultiView'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
