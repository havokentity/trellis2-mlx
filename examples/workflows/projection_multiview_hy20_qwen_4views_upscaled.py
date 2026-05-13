"""Mirror of upstream ``Projection_MultiView_Hy2.0_Qwen_4Views_Upscaled.json``.

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
        prog='projection_multiview_hy20_qwen_4views_upscaled', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'Projection_MultiView_Hy20_Qwen_4Views_Upscaled.glb')
    parser.add_argument('--views', type=int, default=4, help='Number of views')
    parser.add_argument('--texture-resolution', type=int, default=4096, help='UV atlas size')
    return not_implemented_stub(
        parser, workflow_json='Projection_MultiView_Hy2.0_Qwen_4Views_Upscaled.json',
        missing_features=['MultiViewTexturing pipeline (upscaled)', 'Hunyuan3D + Qwen 4-view'],
        upstream_nodes=['Trellis2MeshTexturing', 'Trellis2MultiViewTexturing'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
