"""End-to-end image-to-3D inference pipeline.

Implements the orchestration described in ``PHASE0_SPEC.md §2`` (steps 1–9)
and ``reference/microsoft-trellis2/trellis2/pipelines/trellis2_image_to_3d.py``.

Current state: **shape-only** path is wired up (steps 1-6, 8 of the spec).
Material decoding (step 7) lands once the material decoder is ported.

Pipeline_type ``"512"`` (the simplest) flow:

1. Image preprocessing — RGBA crop + ImageNet normalize.
2. DINOv3-L forward on the image at 512×512 → image conditioning tokens.
3. Stage 1 SS-DiT sampler → ``[1, 8, 16, 16, 16]`` SS latent.
4. SS-VAE decoder → ``[1, 1, 64, 64, 64]`` occupancy → threshold > 0,
   max-pool to ``[1, 1, 32, 32, 32]`` → nonzero coords ``[L, 3]``.
5. Stage 2 SLAT shape DiT sampler → ``[L, 32]`` shape latent.
6. Denormalize (per-channel mean / std from pipeline.json).
7. SC-VAE shape decoder → fine O-Voxel at ``512³``.
8. Flexible Dual Grid mesh extraction → triangles.

Per-stage CFG / step / interval params come straight from
``reference/weights/pipeline.json``.

BiRefNet (background removal) is **not yet wired up** — if the input image
has a non-trivial alpha channel, we use it; otherwise the caller is
expected to pre-mask the input.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from trellis2_mlx.models.dinov3 import DINOv3L, DINOv3LConfig
from trellis2_mlx.models.dit import (
    SLatFlowConfig,
    SLatFlowModel,
    SparseStructureFlowConfig,
    SparseStructureFlowModel,
)
from trellis2_mlx.models.vae import (
    MaterialDecoder,
    MaterialDecoderConfig,
    ShapeDecoder,
    ShapeDecoderConfig,
    SparseStructureDecoder,
    SparseStructureDecoderConfig,
)
from trellis2_mlx.ovoxel.mesh_extract import extract_mesh
from trellis2_mlx.ovoxel.postprocess import export_glb
from trellis2_mlx.samplers.rectified_flow import RectifiedFlowSampler, SamplerParams
from trellis2_mlx.utils.weight_convert import (
    dinov3_from_pt_state_dict,
    material_decoder_from_pt_state_dict,
    shape_decoder_from_pt_state_dict,
    slat_flow_model_from_pt_state_dict,
    ss_decoder_from_pt_state_dict,
    ss_flow_model_from_pt_state_dict,
)

# pipeline.json defaults — match microsoft/TRELLIS.2-4B/pipeline.json exactly.
_SS_SAMPLER_PARAMS = SamplerParams(
    steps=12,
    guidance_strength=7.5,
    guidance_rescale=0.7,
    guidance_interval=(0.6, 1.0),
    rescale_t=5.0,
)
_SHAPE_SAMPLER_PARAMS = SamplerParams(
    steps=12,
    guidance_strength=7.5,
    guidance_rescale=0.5,
    guidance_interval=(0.6, 1.0),
    rescale_t=3.0,
)
_TEX_SAMPLER_PARAMS = SamplerParams(
    steps=12,
    guidance_strength=1.0,  # texture stage effectively turns CFG off
    guidance_rescale=0.0,
    guidance_interval=(0.6, 0.9),
    rescale_t=3.0,
)


def _load_full_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load every tensor in a safetensors file as fp32 numpy. Handles fp16 + bf16."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode())
        data_start = 8 + n
    header.pop("__metadata__", None)
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        for k, info in header.items():
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            buf = f.read(end - start)
            if info["dtype"] == "BF16":
                raw = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
                arr = raw.view(np.float32).reshape(info["shape"]).copy()
            elif info["dtype"] == "F16":
                arr = (
                    np.frombuffer(buf, dtype=np.float16)
                    .reshape(info["shape"])
                    .copy()
                    .astype(np.float32)
                )
            elif info["dtype"] == "F32":
                arr = np.frombuffer(buf, dtype=np.float32).reshape(info["shape"]).copy()
            else:
                raise ValueError(f"unsupported dtype {info['dtype']} for {k}")
            out[k] = arr
    return out


@dataclass
class Trellis2Config:
    """Static configuration for a Trellis2 image-to-3D run.

    Only ``"512"`` is supported today (32³ SLAT → 512³ output). The cascade
    and 1024 modes need extra wiring (the upsample step in the pipeline +
    the 1024 SLAT DiT) — they land once we have a working 512 path to
    compare against.
    """

    pipeline_type: str = "512"
    weights_root: Path = Path("reference/weights")
    ss_decoder_path: Path = Path("reference/weights/trellis-1/ss_dec_conv3d_16l8_fp16.safetensors")
    seed: int = 0
    # When True, also runs the texture SLAT DiT + material decoder and bakes
    # per-vertex colors into the GLB. Adds ~2× wall-time. When False, only
    # geometry (no per-vertex color) is produced. Default ON.
    with_texture: bool = True


@dataclass
class Trellis2ImageTo3DResult:
    """End-to-end pipeline output."""

    vertices: mx.array  # [V, 3] mesh vertex positions
    faces: mx.array  # [F, 3] int32 triangle indices
    active_coords: mx.array  # [L_fine, 3] fine-grid voxel coords
    coarse_coords: mx.array  # [L_coarse, 3] SS-active coords at SLAT resolution
    shape_slat: mx.array  # [L_coarse, 32] denormalized shape latent
    output_resolution: int
    # When the texture pipeline ran, per-vertex PBR material attributes
    # (one row per ``vertices`` row). ``None`` if texture was disabled.
    vertex_colors: mx.array | None = None  # [V, 3] linear-space RGB in [0, 1]
    vertex_metallic: mx.array | None = None  # [V, 1] in [0, 1]
    vertex_roughness: mx.array | None = None  # [V, 1] in [0, 1]
    vertex_alpha: mx.array | None = None  # [V, 1] in [0, 1]


class Trellis2ImageTo3DPipeline:
    """Top-level image-to-3D pipeline. See ``PHASE0_SPEC.md §2``.

    Loads every model once at construction; ``run(image, seed=...)``
    samples one shape from the conditioning image and returns the
    extracted mesh.

    Memory: at fp32 promotion, the loaded models total ~5 GB:

    * DINOv3-L          ~ 300 M params
    * SS-DiT            ~ 1.3 B
    * SS-VAE decoder    ~  37 M
    * Shape-SLAT-DiT    ~ 1.3 B
    * SC-VAE shape dec  ~ 474 M

    On M4 Max with 36 GB unified memory this fits comfortably; on
    smaller systems you may need to lazy-load (TODO).
    """

    def __init__(self, cfg: Trellis2Config | None = None) -> None:
        self.cfg = cfg or Trellis2Config()
        if self.cfg.pipeline_type != "512":
            raise NotImplementedError(
                f"pipeline_type={self.cfg.pipeline_type!r} not yet supported; "
                "only '512' is wired up. 1024 / 1024_cascade / 1536_cascade land next."
            )

        # Stage-1 latent grid = 16³; SLAT latent grid = 32³; output = 512³.
        self._ss_latent_res = 16
        self._slat_res = 32
        self._output_res = 512

        # Models — instantiated and weight-loaded.
        self._dinov3 = self._load_dinov3()
        self._ss_dit = self._load_ss_dit()
        self._ss_decoder = self._load_ss_decoder()
        self._shape_dit = self._load_shape_dit()
        self._shape_decoder = self._load_shape_decoder()
        # Texture-side models are optional.
        self._tex_dit: SLatFlowModel | None = None
        self._material_decoder: MaterialDecoder | None = None
        if self.cfg.with_texture:
            self._tex_dit = self._load_tex_dit()
            self._material_decoder = self._load_material_decoder()

        # Sampler is stateless apart from sigma_min.
        self._sampler = RectifiedFlowSampler(sigma_min=1e-5)

        # SLAT denormalization stats from pipeline.json.
        pipeline_json = json.loads((self.cfg.weights_root / "pipeline.json").read_text())
        norm = pipeline_json["args"]["shape_slat_normalization"]
        self._shape_slat_mean = mx.array(norm["mean"], dtype=mx.float32)
        self._shape_slat_std = mx.array(norm["std"], dtype=mx.float32)
        tex_norm = pipeline_json["args"]["tex_slat_normalization"]
        self._tex_slat_mean = mx.array(tex_norm["mean"], dtype=mx.float32)
        self._tex_slat_std = mx.array(tex_norm["std"], dtype=mx.float32)

    # ── model loaders ──────────────────────────────────────────────────

    def _load_dinov3(self) -> DINOv3L:
        from transformers import DINOv3ViTModel

        # We pull from transformers because the published TRELLIS.2-4B repo
        # doesn't bundle the DINOv3 weights — they live at facebook/...
        pt_model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval()  # type: ignore[no-untyped-call]
        state = {k: v.detach().cpu().numpy() for k, v in pt_model.state_dict().items()}
        # The pipeline calls DINOv3 at 512 (cond_512). 1024 only matters for
        # the 1024 SLAT path which we haven't wired yet.
        cfg = DINOv3LConfig(image_size=512)
        model = DINOv3L(cfg)
        model.load_weights(
            dinov3_from_pt_state_dict(state, num_hidden_layers=cfg.num_hidden_layers)
        )
        return model

    def _load_ss_dit(self) -> SparseStructureFlowModel:
        path = self.cfg.weights_root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
        state = _load_full_safetensors(path)
        model = SparseStructureFlowModel(SparseStructureFlowConfig())
        model.load_weights(ss_flow_model_from_pt_state_dict(state))
        return model

    def _load_ss_decoder(self) -> SparseStructureDecoder:
        state = _load_full_safetensors(self.cfg.ss_decoder_path)
        model = SparseStructureDecoder(SparseStructureDecoderConfig())
        model.load_weights(ss_decoder_from_pt_state_dict(state))
        return model

    def _load_shape_dit(self) -> SLatFlowModel:
        path = self.cfg.weights_root / "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
        state = _load_full_safetensors(path)
        model = SLatFlowModel(SLatFlowConfig(resolution=self._slat_res))
        model.load_weights(slat_flow_model_from_pt_state_dict(state))
        return model

    def _load_shape_decoder(self) -> ShapeDecoder:
        path = self.cfg.weights_root / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
        state = _load_full_safetensors(path)
        model = ShapeDecoder(ShapeDecoderConfig())
        model.load_weights(shape_decoder_from_pt_state_dict(state))
        return model

    def _load_tex_dit(self) -> SLatFlowModel:
        # Same architecture as shape SLAT DiT but with in_channels=64 (the
        # texture DiT receives the shape latent concatenated channel-wise).
        path = self.cfg.weights_root / "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"
        state = _load_full_safetensors(path)
        model = SLatFlowModel(SLatFlowConfig(resolution=self._slat_res, in_channels=64))
        model.load_weights(slat_flow_model_from_pt_state_dict(state))
        return model

    def _load_material_decoder(self) -> MaterialDecoder:
        path = self.cfg.weights_root / "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"
        state = _load_full_safetensors(path)
        model = MaterialDecoder(MaterialDecoderConfig())
        model.load_weights(material_decoder_from_pt_state_dict(state))
        return model

    # ── preprocessing ──────────────────────────────────────────────────

    @staticmethod
    def _preprocess_image(image: Image.Image, target_size: int = 512) -> mx.array:
        """Crop to subject (using alpha if present), resize, ImageNet-normalize.

        Mirrors ``trellis2_image_to_3d.py:127-162``. If the image has a
        non-trivial alpha channel, we crop to the subject's bbox at α > 0.8.
        Otherwise we assume the caller pre-masked the image (BiRefNet
        wrapper not yet in).
        """
        if image.mode == "RGBA":
            arr = np.asarray(image)
            alpha = arr[:, :, 3]
            if not np.all(alpha == 255):
                bbox_pts = np.argwhere(alpha > 0.8 * 255)
                if bbox_pts.size > 0:
                    y0, x0 = bbox_pts.min(axis=0)
                    y1, x1 = bbox_pts.max(axis=0)
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    size = max(x1 - x0, y1 - y0)
                    half = size // 2
                    image = image.crop((cx - half, cy - half, cx + half + 1, cy + half + 1))
                # Composite onto black using alpha (premultiplied)
                arr = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
                rgb = arr[:, :, :3] * arr[:, :, 3:4]
                image = Image.fromarray((rgb * 255).astype(np.uint8))

        image = image.convert("RGB").resize((target_size, target_size), Image.LANCZOS)
        arr = np.asarray(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return mx.array(arr[None])  # NHWC with batch=1

    # ── pipeline steps ─────────────────────────────────────────────────

    def _encode_image(self, image: Image.Image) -> mx.array:
        """Run DINOv3-L, return ``[B, N, 1024]`` patch features."""
        pixels = self._preprocess_image(image, 512)
        cond = self._dinov3(pixels)
        return cond

    def _sample_ss_latent(
        self, cond: mx.array, neg_cond: mx.array, rng: np.random.Generator
    ) -> mx.array:
        """Sample the stage-1 SS latent. Output ``[1, 8, 16, 16, 16]``."""
        r = self._ss_latent_res
        noise = mx.array(rng.standard_normal((1, 8, r, r, r)).astype(np.float32))

        def model_fn(x: mx.array, t_scaled: mx.array, c: mx.array, **kw: Any) -> mx.array:
            return self._ss_dit(x, t_scaled, c)

        return self._sampler.sample(
            model_fn, noise, cond=cond, neg_cond=neg_cond, params=_SS_SAMPLER_PARAMS
        )

    def _decode_ss_latent(self, z_s: mx.array) -> mx.array:
        """SS latent → active coords at SLAT resolution.

        Steps from trellis2_image_to_3d.py:223-235:

        1. occupancy = ss_decoder(z_s) > 0      # [1, 1, 64, 64, 64] bool
        2. max-pool to slat_res (32 here, ratio 2) > 0.5  → [1, 1, 32, 32, 32]
        3. argwhere → [L, 4] (b, z, y, x); drop batch → [L, 3].
        """
        decoded = self._ss_decoder(z_s)  # [1, 1, 64, 64, 64]
        occ = decoded > 0
        target_res = self._slat_res
        decoded_res = occ.shape[-1]  # 64 in NCDHW; or NDHWC's 4th dim
        ratio = decoded_res // target_res
        if ratio > 1:
            # Max-pool over each (ratio, ratio, ratio) block. occ is NCDHW.
            # Use MLX's MaxPool3d on NDHWC then transpose, or just do it with
            # reshape since ratio is small.
            occ_f = occ.astype(mx.float32)
            # NCDHW [1, 1, 64, 64, 64] → [1, 1, 32, 2, 32, 2, 32, 2] → max over axes 3,5,7
            b, c, d, h, w = occ_f.shape
            occ_f = occ_f.reshape(b, c, target_res, ratio, target_res, ratio, target_res, ratio)
            occ_f = mx.max(occ_f, axis=(3, 5, 7))
            occ = occ_f > 0.5
        # Argwhere via numpy
        occ_np = np.asarray(occ).reshape(target_res, target_res, target_res)
        zyx = np.argwhere(occ_np).astype(np.int32)  # [L, 3]
        return mx.array(zyx)

    def _sample_shape_slat(
        self,
        coords: mx.array,
        cond: mx.array,
        neg_cond: mx.array,
        rng: np.random.Generator,
    ) -> mx.array:
        """Sample the SLAT shape latent on the given active set. Output ``[L, 32]``."""
        n_active = coords.shape[0]
        noise = mx.array(rng.standard_normal((n_active, 32)).astype(np.float32))

        def model_fn(x: mx.array, t_scaled: mx.array, c: mx.array, **kw: Any) -> mx.array:
            return self._shape_dit(x, kw["coords"], t_scaled, c)

        slat = self._sampler.sample(
            model_fn,
            noise,
            cond=cond,
            neg_cond=neg_cond,
            params=_SHAPE_SAMPLER_PARAMS,
            coords=coords,
        )
        # Denormalize: slat = slat * std + mean  (per-channel)
        slat = slat * self._shape_slat_std + self._shape_slat_mean
        return slat

    def _sample_tex_slat(
        self,
        coords: mx.array,
        shape_slat_denormalized: mx.array,
        cond: mx.array,
        neg_cond: mx.array,
        rng: np.random.Generator,
    ) -> mx.array:
        """Sample the texture SLAT latent. Output ``[L, 32]`` denormalized.

        The texture DiT receives ``concat_cond = shape_slat_normalized`` —
        the shape latent is RE-normalized (using the shape stats) before
        being concatenated channel-wise to the texture noise per
        ``trellis2_image_to_3d.py:407-419``. After sampling, we apply the
        *texture* denormalization stats to produce the actual texture
        latent.
        """
        if self._tex_dit is None:
            raise RuntimeError("texture pipeline disabled (cfg.with_texture=False)")
        n_active = coords.shape[0]
        # Re-normalize shape latent for use as concat_cond.
        shape_slat_norm = (shape_slat_denormalized - self._shape_slat_mean) / self._shape_slat_std
        noise = mx.array(rng.standard_normal((n_active, 32)).astype(np.float32))

        def model_fn(x: mx.array, t_scaled: mx.array, c: mx.array, **kw: Any) -> mx.array:
            return self._tex_dit(x, kw["coords"], t_scaled, c, concat_cond=kw["concat_cond"])

        tex = self._sampler.sample(
            model_fn,
            noise,
            cond=cond,
            neg_cond=neg_cond,
            params=_TEX_SAMPLER_PARAMS,
            coords=coords,
            concat_cond=shape_slat_norm,
        )
        # Denormalize with the texture stats.
        return tex * self._tex_slat_std + self._tex_slat_mean

    # ── public API ─────────────────────────────────────────────────────

    def run(
        self,
        image: Image.Image,
        *,
        seed: int | None = None,
    ) -> Trellis2ImageTo3DResult:
        """Run the full image-to-3D pipeline.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image. For best results pre-mask with BiRefNet or
            provide an RGBA image with a clean alpha channel.
        seed : int or None
            RNG seed. Defaults to ``self.cfg.seed``.

        Returns
        -------
        Trellis2ImageTo3DResult
            Final mesh + intermediate latents.
        """
        import sys

        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)

        # 1. Image conditioning. clear_cache() between every stage frees any
        # transient MLX buffers and keeps peak GPU memory near the stage's
        # working set rather than the cumulative sum.
        print("  [1/5] DINOv3 encoding ...", flush=True, file=sys.stderr)
        cond = self._encode_image(image)
        mx.eval(cond)
        mx.clear_cache()
        neg_cond = mx.zeros_like(cond)
        mx.eval(neg_cond)

        # 2. Sample SS latent
        print(
            "  [2/5] SS-DiT sampling (12 steps × 2 CFG branches) ...", flush=True, file=sys.stderr
        )
        z_s = self._sample_ss_latent(cond, neg_cond, rng)
        mx.eval(z_s)
        mx.clear_cache()

        # 3. SS-VAE decode → active coords
        print("  [3/5] SS-VAE decoder + maxpool ...", flush=True, file=sys.stderr)
        coords = self._decode_ss_latent(z_s)
        mx.eval(coords)
        mx.clear_cache()
        if coords.shape[0] == 0:
            raise RuntimeError(
                "SS-VAE decoder produced an empty active set; "
                "the generation collapsed (try a different seed)"
            )

        # 4. Sample shape SLAT
        print(
            f"  [4/5] SLAT-shape DiT sampling on {coords.shape[0]} voxels ...",
            flush=True,
            file=sys.stderr,
        )
        slat = self._sample_shape_slat(coords, cond, neg_cond, rng)
        mx.eval(slat)
        # Round-trip slat / coords through numpy so any MLX graph references
        # tied to upstream computations are fully cut before the SC-VAE
        # decoder runs. This noticeably lowers peak memory.
        slat = mx.array(np.asarray(slat))
        coords = mx.array(np.asarray(coords))
        mx.clear_cache()

        # 5. SC-VAE shape decoder
        n_steps = "5/7" if self.cfg.with_texture else "5/5"
        print(
            f"  [{n_steps}] SC-VAE shape decoder + mesh extraction ...",
            flush=True,
            file=sys.stderr,
        )
        ovoxel = self._shape_decoder(slat, coords, coarse_resolution=self._slat_res)
        mx.eval(ovoxel.coords, ovoxel.v, ovoxel.delta_logits, ovoxel.gamma)
        for sl in ovoxel.subdiv_logits:
            mx.eval(sl)
        mx.clear_cache()

        # 6. Mesh extraction
        verts, faces = extract_mesh(
            ovoxel.coords,
            ovoxel.v,
            ovoxel.delta_logits,
            ovoxel.gamma,
            grid_size=ovoxel.output_resolution,
        )
        mx.eval(verts, faces)

        # 7. (optional) texture pipeline: sample texture SLAT, run material
        # decoder with the shape decoder's subdivisions, bake per-vertex color.
        vertex_colors: mx.array | None = None
        vertex_metallic: mx.array | None = None
        vertex_roughness: mx.array | None = None
        vertex_alpha: mx.array | None = None
        if self.cfg.with_texture:
            print(
                "  [6/7] SLAT-texture DiT sampling (concat_cond = shape latent) ...",
                flush=True,
                file=sys.stderr,
            )
            tex_slat = self._sample_tex_slat(coords, slat, cond, neg_cond, rng)
            mx.eval(tex_slat)
            tex_slat = mx.array(np.asarray(tex_slat))  # cut graph
            mx.clear_cache()

            print(
                "  [7/7] Material decoder + per-vertex color baking ...",
                flush=True,
                file=sys.stderr,
            )
            # Binarize shape's subdiv logits to drive the material decoder's
            # upsamples (matches upstream `decode_tex_slat(..., guide_subs=subs)`).
            guide_subs = [sl > 0 for sl in ovoxel.subdiv_logits]
            mat = self._material_decoder(
                tex_slat, coords, coarse_resolution=self._slat_res, guide_subs=guide_subs
            )
            mx.eval(mat.coords, mat.base_color, mat.metallic, mat.roughness, mat.alpha)
            mx.clear_cache()

            # Vertex-color baking: extract_mesh produced one dual vertex per
            # active voxel, in the same order as ovoxel.coords. The material
            # decoder runs with the same guide_subs, so its output coords
            # match the shape decoder's coords (and therefore the vertex
            # indices) row-for-row. Hence per-vertex color is simply the
            # material decoder's per-voxel base_color.
            if mat.coords.shape != ovoxel.coords.shape:
                raise RuntimeError(
                    f"material decoder produced {mat.coords.shape[0]} voxels but "
                    f"shape decoder produced {ovoxel.coords.shape[0]} — "
                    "guide_subs alignment broke"
                )
            vertex_colors = mat.base_color
            vertex_metallic = mat.metallic
            vertex_roughness = mat.roughness
            vertex_alpha = mat.alpha

        return Trellis2ImageTo3DResult(
            vertices=verts,
            faces=faces,
            active_coords=ovoxel.coords,
            coarse_coords=coords,
            shape_slat=slat,
            output_resolution=ovoxel.output_resolution,
            vertex_colors=vertex_colors,
            vertex_metallic=vertex_metallic,
            vertex_roughness=vertex_roughness,
            vertex_alpha=vertex_alpha,
        )

    def export_glb(
        self,
        result: Trellis2ImageTo3DResult,
        out_path: str | Path,
        *,
        repair: bool = True,
    ) -> Path:
        """Write the result mesh to a GLB file. If the result has per-vertex
        colors (from the texture pipeline), they're authored as the GLB's
        vertex-color attribute and show up in any glTF viewer.

        ``repair=True`` (default) runs trimesh's normal-fixing pass on each
        connected component so back-facing triangles flip outward. Pass
        ``repair=False`` to skip that and emit the raw extractor output."""
        return export_glb(
            result.vertices,
            result.faces,
            out_path,
            material_colors=result.vertex_colors,
            repair=repair,
        )


__all__ = ["Trellis2Config", "Trellis2ImageTo3DPipeline", "Trellis2ImageTo3DResult"]
