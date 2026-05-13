# Workflow scripts — one Python file per upstream ComfyUI workflow

This folder mirrors every JSON in upstream
[`ComfyUI-Trellis2/example_workflows/`](https://github.com/visualbruno/ComfyUI-Trellis2/tree/main/example_workflows)
as a standalone Python script with matching CLI parameters. Each script
either:

* **Runs end-to-end** on Apple Silicon via MLX (✅ rows below), or
* **Prints a structured "not yet implemented" report** listing the
  upstream nodes / features it needs (⏳ rows below).

Run any script with `--help` to see its parameters. Every script
defaults to the upstream sample image and seed=0.

## Support matrix

| Workflow | Script | Status | Notes |
|---|---|---|---|
| `Simple.json` | `simple.py` | ✅ runs all modes | 512 / 1024_cascade / 1536_cascade |
| `LowPoly.json` | `low_poly.py` | ✅ runs at 512 (native) | Default target_faces=5000 |
| `Only_Mesh_Simple.json` | `only_mesh_simple.py` | ✅ runs all modes | Mesh only |
| `Only_Mesh_Advanced.json` | `only_mesh_advanced.py` | ✅ runs all modes | Mesh only, advanced sampler knobs |
| `Better_Texture.json` | `better_texture.py` | ✅ runs all modes | Vertex-color texture |
| `Advanced.json` | `advanced.py` | ✅ runs all modes | Full advanced sampler |
| `Using_Qwen_Rembg.json` | `using_qwen_rembg.py` | ✅ utility | Reports alpha stats |
| `Max_Quality.json` | `max_quality.py` | ⏳ stub | Needs MeshRefiner + 4K texture atlas (1536_cascade now works) |
| `High_Quality.json` | `high_quality.py` | ⏳ stub | Needs MeshRefiner (1024_cascade now works) |
| `MeshRefiner.json` | `mesh_refiner.py` | ⏳ stub | Needs MeshRefiner DiT model |
| `RefineMesh.json` | `refine_mesh.py` | ⏳ stub | Needs MeshRefiner + Trellis2LoadMesh |
| `RefineMesh_MeshOnly.json` | `refine_mesh_mesh_only.py` | ⏳ stub | Needs MeshRefiner + Trellis2LoadMesh |
| `TextureMesh.json` | `texture_mesh.py` | ⏳ stub | Needs texture-on-arbitrary-mesh + UV atlas |
| `MultiViews.json` | `multi_views.py` | ⏳ stub | Needs MultiView generator |
| `MultiViews_MeshOnly.json` | `multi_views_mesh_only.py` | ⏳ stub | Needs MultiView generator |
| `MultiViews_TextureMesh.json` | `multi_views_texture_mesh.py` | ⏳ stub | Needs MultiViewTexturing |
| `Advanced_CustomSteps.json` | `advanced_custom_steps.py` | ⏳ stub | Needs decomposed step generators |
| `Advanced_CustomSteps_MeshOnly.json` | `advanced_custom_steps_mesh_only.py` | ⏳ stub | Needs decomposed step generators |
| `ReconViaGen_MeshOnly.json` | `recon_via_gen.py` | ⏳ stub | Needs ReconViaGen sparse generator |
| `ReconViaGen_MeshOnly_FromVideo.json` | `recon_via_gen_video.py` | ⏳ stub | Needs ReconViaGen + video frame extraction |
| `Watertight_Mesh.json` | `watertight_mesh.py` | ⏳ stub | Needs VoxelToMesh + TexSlat |
| `Watertight_No_Holes.json` | `watertight_no_holes.py` | ⏳ stub | Needs VoxelToMesh + TexSlat + MeshRefiner |
| `Projection_6Views_Hy20.json` | `projection_6views_hy20.py` | ⏳ stub | Needs Hunyuan3D 2.0 + MultiViewTexturing |
| `Projection_MultiView_Hy2.0_Qwen_2Views.json` | `projection_multiview_hy20_qwen_2views.py` | ⏳ stub | Needs Hunyuan3D + Qwen + MultiViewTexturing |
| `Projection_MultiView_Hy2.0_Qwen_2Views_LowPoly.json` | `projection_multiview_hy20_qwen_2views_lowpoly.py` | ⏳ stub | " |
| `Projection_MultiView_Hy2.0_Qwen_2Views_LowPoly_FullWorkflow.json` | `projection_multiview_hy20_qwen_2views_lowpoly_full.py` | ⏳ stub | " |
| `Projection_MultiView_Hy2.0_Qwen_2Views_LowPoly_FullWorkflow_Fast.json` | `projection_multiview_hy20_qwen_2views_lowpoly_full_fast.py` | ⏳ stub | " |
| `Projection_MultiView_Hy2.0_Qwen_2Views_Upscaled.json` | `projection_multiview_hy20_qwen_2views_upscaled.py` | ⏳ stub | " |
| `Projection_MultiView_Hy2.0_Qwen_4Views.json` | `projection_multiview_hy20_qwen_4views.py` | ⏳ stub | " |
| `Projection_MultiView_Hy2.0_Qwen_4Views_Upscaled.json` | `projection_multiview_hy20_qwen_4views_upscaled.py` | ⏳ stub | " |
| `Projection_MultiView_with_Hunyuan3D2.0.json` | `projection_multiview_with_hunyuan3d.py` | ⏳ stub | " |

**Tally:** 7 / 31 workflows runnable today; 24 stubs awaiting model
ports. All cascade modes (`1024_cascade`, `1536_cascade`) now run
natively on the 7 implemented workflows.

## Quickstart

```bash
# Fully textured mesh at 1024_cascade (3.7× more verts than 512 mode)
uv run python -m examples.workflows.advanced --mode 1024_cascade

# Highest detail — 1536_cascade
uv run python -m examples.workflows.simple --mode 1536_cascade --target-faces 1000000

# Game-ready low-poly with a proper UV-baked 1K texture atlas
uv run python -m examples.workflows.low_poly --target-faces 5000 \
    --uv-atlas --texture-size 1024

# Geometry only, fast
uv run python -m examples.workflows.only_mesh_simple \
    --mode 512 --target-faces 100000

# Your own image, cascade + UV atlas at 2K
uv run python -m examples.workflows.advanced \
    --image path/to/your.png --output ./out.glb \
    --mode 1024_cascade --uv-atlas --texture-resolution 2048

# See what an unimplemented workflow would need
uv run python -m examples.workflows.max_quality
```

### UV-atlas baking (`--uv-atlas`)

By default the implemented workflows ship per-vertex colors. With
`--uv-atlas`, the per-vertex colors are baked into a 2D texture atlas
via `xatlas` (LSCM-style chart unwrap) + a software rasterizer.
Output GLBs carry a proper `PBRMaterial` with `baseColorTexture` and
optional `metallicRoughnessTexture` so any glTF viewer / game engine
samples a real texture rather than interpolating per-vertex colors.

The atlas pass adds 10–60 s depending on `--texture-size` and target
poly count. For aggressive decimation (e.g. 5k faces) it's almost
free and dramatically improves visual quality.

Output GLB files default to the repository root with the workflow name
(e.g. `Simple.glb`, `LowPoly.glb`).

**Wall time (on M4 Max):**

| Mode | Generation | Notes |
|---|---|---|
| `512` | ~70 s | Default, ~1.15M raw verts |
| `1024_cascade` | ~230 s | ~4.3M raw verts (3.7× more) |
| `1536_cascade` | ~500 s (est.) | ~9-12M raw verts (rough est.) |

Post-processing (`fix_normals` + decimation) adds roughly 1.5×
generation time at 1024_cascade.

## What's missing — the porting roadmap

Each ⏳ row above maps to a concrete piece of upstream we need to port:

1. **Cascade generators** (`1024_cascade`, `1536_cascade`) — the highest
   single-leverage quality jump. Re-uses the existing SS/SLAT DiT
   architecture at finer voxel resolutions; need new weights + a new
   sparse-conv level wired into the SC-VAE decoders.
2. **MeshRefiner DiT** — second-pass image-conditioned geometry
   refinement. Separate set of weights.
3. **MultiViewGenerator + MultiViewTexturing** — multi-image input
   pathway. The DINOv3 frontend extends straightforwardly; the harder
   piece is the MV-texturing camera-projection raster bake.
4. **TexSlatGenerator + VoxelToMesh** — used by Watertight workflows.
   `VoxelToMesh` is a closed-surface extractor (vs our FDG which can
   produce open patches); `TexSlat` is a decomposed texture stage.
5. **UV unwrap + texture atlas** — needed for proper textures vs the
   per-vertex colors we ship today. Apple-side candidates: `xatlas` +
   custom Metal raster pass (since `nvdiffrast` is CUDA-only).
6. **Hunyuan3D 2.0 / Qwen integration** — these are external models not
   in trellis2-mlx's scope per CLAUDE.md, but the projection workflows
   could optionally call out to them.
7. **ReconViaGen sparse generator** — a variant of the SLAT DiT that
   reconstructs from generated views.
8. **Video frame extraction** — trivial wrapper around `opencv`/`pyav`.

## Running tests against a workflow script

The implemented workflows can be smoke-tested by piping their output
through trimesh to verify the GLB is well-formed:

```bash
uv run python -m examples.workflows.low_poly --target-faces 5000
uv run python -c "
import trimesh
m = trimesh.load('LowPoly.glb', force='mesh')
print(f'verts={m.vertices.shape[0]}  faces={m.faces.shape[0]}')
"
```
