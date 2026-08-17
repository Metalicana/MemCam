import os
import json
import math
import torch

import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F

from dataset.poses import compute_relative_pose

from .base import BasePipeline
from .memory_policies import (
    FrameMemoryBuffer,
    SurpriseForcingMemoryController,
    VisualMemoryFeatureExtractor,
    compute_density_balanced_view_coverage_scores,
    compute_facility_coreset_scores,
    compute_future_view_coverage_scores,
    compute_h2o_heavy_hitter_scores,
    compute_kcenter_coreset_scores,
    compute_marginal_coverage_eviction_scores,
    compute_rarity_irreplaceability_scores,
    compute_slam_covisibility_scores,
    compute_slam_max_coverage_scores,
    compute_slam_ri_blend_scores,
    compute_trajectory_coverage_scores,
    image_quality_scores_from_pil_images,
)
from .memory_profiling import (
    MemoryRolloutProfiler,
    numpy_mapping_nbytes,
    tensor_mapping_nbytes,
)
from .memory_attention_audit import (
    MemoryAttentionCollector,
    TargetValueDescriptorCollector,
    add_retrieval_controls,
    select_intervention_candidates,
)
from ..prompters import WanPrompter
from ..schedulers.flow_match import FlowMatchScheduler

from ..models import ModelManager
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear

from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_overlap import calculate_overlap_from_c2w
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample 
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm

from utils.compressor_utils import pad_for_3d_conv, compute_context_rope

TARGET_LENGTH = 20  # anchor 1l + predict 19l
ANCHOR_LENGTH = 1   # anchor 4f 1l or 1f 1l
PREDICT_FRAMES = 76 # 76 frames to predict per section
FRAMES_PER_SECTION = 77 # section 77f 20l

FOV_HALF_H = 45.0    # 水平半视场角（度）增大→更宽松的重叠判定
FOV_HALF_V = 30.0    # 垂直半视场角（度）增大→更宽松的重叠判定
FOV_SAMPLES = 5000   # 采样点数 增大→更准确但更慢
FOV_RADIUS = 50.0    # 采样球体半径
VISUAL_MEMORY_POLICIES = {
    "rarity_irreplaceability",
    "slam_covisibility",
    "slam_max_coverage",
    "slam_ri_blend",
    "facility_coreset",
    "kcenter_coreset",
    "density_balanced_view_coverage",
    "future_view_coverage",
    "mce",
}
ARCHIVE_MEMORY_POLICIES = {
    "facility_coreset",
    "kcenter_coreset",
    "trajectory_coverage",
}
CORESET_ARCHIVE_STRIDE = 4
CORESET_MAX_ARCHIVE_SIZE = 5000


def normalize_context_selection_overrides(overrides):
    """Normalize replay overrides to ``(section, target) -> metadata``."""
    normalized = {}
    for raw_section, target_rows in (overrides or {}).items():
        section_idx = int(raw_section)
        if section_idx <= 0:
            raise ValueError("Context overrides are only valid after section 0")
        if not isinstance(target_rows, dict):
            raise ValueError("Each context-override section must map targets to frames")
        for raw_target, raw_value in target_rows.items():
            target_frame = int(raw_target)
            if isinstance(raw_value, dict):
                if "memory_frame" not in raw_value:
                    raise ValueError("Context-override metadata requires memory_frame")
                value = dict(raw_value)
                value["memory_frame"] = int(value["memory_frame"])
            else:
                value = {"memory_frame": int(raw_value)}
            normalized[(section_idx, target_frame)] = value
    return normalized


class WanVideoMemCamPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'image_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype, 
                offload_device="cpu",
                onload_dtype=dtype, 
                onload_device="cpu",
                computation_dtype=self.torch_dtype, 
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype, 
                offload_device="cpu",
                onload_dtype=dtype, 
                onload_device=self.device,
                computation_dtype=self.torch_dtype, 
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype, 
                offload_device="cpu",
                onload_dtype=dtype, 
                onload_device="cpu",
                computation_dtype=self.torch_dtype, 
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype, 
                offload_device="cpu",
                onload_dtype=dtype, 
                onload_device=self.device,
                computation_dtype=self.torch_dtype, 
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype, 
                    offload_device="cpu",
                    onload_dtype=dtype, 
                    onload_device="cpu",
                    computation_dtype=dtype, 
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()
    

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoMemCamPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames


    def frame_tensor_to_pil(self, frame_tensor):
        frame_tensor = frame_tensor.detach().float().cpu()
        if frame_tensor.ndim == 5:
            frame_tensor = frame_tensor[0, :, 0]
        elif frame_tensor.ndim == 4:
            frame_tensor = frame_tensor[:, 0]
        frame = ((frame_tensor + 1) * 127.5).clip(0, 255).byte()
        frame = frame.permute(1, 2, 0).numpy()
        return Image.fromarray(frame)
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames


    def forward(
        self,
        context_latents: torch.Tensor,       # (B, C, context_length, H, W)
        target_latents: torch.Tensor,        # (B, C, 20, H, W)
        context_pose: torch.Tensor,          # (B, context_length, 12)
        target_pose: torch.Tensor,           # (B, 20, 12)
        timestep: torch.Tensor,
        context: torch.Tensor,
    ):
        dit = self.dit
        
        # Time embedding
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        
        # Text embedding
        context = dit.text_embedding(context)

        # Context compression
        context_latents_padded = pad_for_3d_conv(context_latents, (1, 4, 4))  # Pad for kernel (1, 4, 4)
        ctx = dit.context_compressor(context_latents_padded)
        f_ctx, h_ctx, w_ctx = ctx.shape[2], ctx.shape[3], ctx.shape[4]
        ctx = rearrange(ctx, 'b c f h w -> b (f h w) c').contiguous()

        # Target patchify
        tgt, (f_tgt, h_tgt, w_tgt) = dit.patchify(target_latents)
        
        # Cat tokens: [context_tokens, target_tokens]
        x = torch.cat([ctx, tgt], dim=1)
        
        # Spatial sizes for cam_emb expansion
        ctx_spatial = h_ctx * w_ctx
        tgt_spatial = h_tgt * w_tgt
        cam_emb = (context_pose, target_pose)
        
        # ========== Context as Memory Style RoPE ==========
        # Target: positions 0-19 (preserve pretrained positions)
        # Context: positions starting from 20 (sequential assignment)
        context_freqs = compute_context_rope(
            dit=dit,
            f_ctx=f_ctx,
            h_tgt=h_tgt, w_tgt=w_tgt,
            h_ctx=h_ctx, w_ctx=w_ctx,
            device=x.device
        )  # (S_ctx, 1, dim) complex
        
        # Target freqs: positions 0 to f_tgt-1 (0-19)
        target_freqs = torch.cat([
            dit.freqs[0][:f_tgt].view(f_tgt, 1, 1, -1).expand(f_tgt, h_tgt, w_tgt, -1),
            dit.freqs[1][:h_tgt].view(1, h_tgt, 1, -1).expand(f_tgt, h_tgt, w_tgt, -1),
            dit.freqs[2][:w_tgt].view(1, 1, w_tgt, -1).expand(f_tgt, h_tgt, w_tgt, -1)
        ], dim=-1).reshape(f_tgt * h_tgt * w_tgt, 1, -1).to(x.device)
        
        # Concatenate: [context_freqs, target_freqs] -> (S_total, 1, dim)
        freqs = torch.cat([context_freqs, target_freqs], dim=0)
        
        # DiT blocks
        for block in dit.blocks:
            x = block(x, context, cam_emb, t_mod, freqs, ctx_spatial=ctx_spatial, tgt_spatial=tgt_spatial)
        
        # Head & unpatchify
        x = dit.head(x, t)
        ctx_tokens = f_ctx * h_ctx * w_ctx
        tgt_tokens = f_tgt * h_tgt * w_tgt
        x_tgt = x[:, ctx_tokens:ctx_tokens + tgt_tokens, :]
        x_tgt = dit.unpatchify(x_tgt, (f_tgt, h_tgt, w_tgt))
        
        return x_tgt


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        c2ws=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=352,
        width=640,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        memory_policy="unbounded",
        memory_budget=None,
        memory_bank_device="cpu",
        density_coverage_alpha=0.5,
        density_coverage_dino_weight=0.5,
        density_coverage_rgb_weight=0.25,
        future_coverage_alpha=0.5,
        future_coverage_dino_weight=0.5,
        future_coverage_rgb_weight=0.25,
        future_coverage_query_stride=19,
        future_coverage_query_weight=1.0,
        mce_alpha=0.65,
        mce_lambda=None,
        mce_gamma=0.25,
        mce_query_stride=19,
        mce_rarity_neighbors=3,
        ri_rarity_neighbors=3,
        slamri_beta=0.5,
        slamri_rarity_neighbors=3,
        surprise_alpha=0.7,
        surprise_ema_momentum=0.95,
        surprise_controller_step=0.1,
        surprise_target_admission_ratio=0.3,
        surprise_initial_threshold=0.002,
        surprise_surprise_weight=1.8,
        surprise_usage_weight=1.0,
        surprise_age_weight=0.4,
        surprise_route_top_k=3,
        surprise_value_layer=15,
        surprise_warmup_sections=3,
        context_selection_overrides=None,
        stop_after_section=None,
        access_trace_path=None,
        access_trace_metadata=None,
        attention_audit_path=None,
        attention_audit_metadata=None,
        attention_probe_sections=None,
        attention_probe_steps=None,
        attention_probe_layers=None,
        attention_query_count=128,
        attention_query_chunk_size=16,
        profile_path=None,
        profile_metadata=None,
        progress_bar_cmd=tqdm
    ):
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Encode Prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
        
        # Sections
        total_frames = c2ws.shape[0]
        print(f"Total frames: {total_frames}")
        assert total_frames % 76 == 1
        total_sections = (total_frames - 1) // 76
        print(f"Total sections: {total_sections}")
        context_selection_overrides = normalize_context_selection_overrides(
            context_selection_overrides
        )
        if stop_after_section is None:
            generated_section_count = total_sections
        else:
            stop_after_section = int(stop_after_section)
            if stop_after_section < 0 or stop_after_section >= total_sections:
                raise ValueError(
                    f"stop_after_section must be in [0, {total_sections - 1}]"
                )
            generated_section_count = stop_after_section + 1
        invalid_override_sections = sorted(
            {
                section_idx
                for section_idx, _target_frame in context_selection_overrides
                if section_idx >= generated_section_count
            }
        )
        if invalid_override_sections:
            raise ValueError(
                "Context overrides refer to sections after the requested stop: "
                f"{invalid_override_sections}"
            )
        print(f"Sections to generate: {generated_section_count}")
        if context_selection_overrides:
            print(
                "Context replay overrides: "
                f"{len(context_selection_overrides)} target selections"
            )

        attention_audit_enabled = attention_audit_path is not None
        probe_sections = {
            int(value) for value in (attention_probe_sections or [])
        }
        probe_steps = {int(value) for value in (attention_probe_steps or [])}
        probe_layers = sorted(
            {int(value) for value in (attention_probe_layers or [])}
        )
        if attention_audit_enabled:
            if not probe_sections or not probe_steps or not probe_layers:
                raise ValueError(
                    "Attention audit requires non-empty probe sections, steps, and layers"
                )
            invalid_sections = sorted(
                value for value in probe_sections if value <= 0 or value >= total_sections
            )
            invalid_steps = sorted(
                value for value in probe_steps if value < 0 or value >= num_inference_steps
            )
            invalid_layers = sorted(
                value for value in probe_layers if value < 0 or value >= len(self.dit.blocks)
            )
            if invalid_sections:
                raise ValueError(f"Invalid attention probe sections: {invalid_sections}")
            if invalid_steps:
                raise ValueError(f"Invalid attention probe steps: {invalid_steps}")
            if invalid_layers:
                raise ValueError(f"Invalid attention probe layers: {invalid_layers}")
            if attention_query_count < 1 or attention_query_chunk_size < 1:
                raise ValueError("Attention query counts and chunk sizes must be positive")

        # Encode input image
        self.load_models_to_device(['vae'])
        input_image_tensor = self.preprocess_image(input_image).permute(1, 0, 2, 3).unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
        start_latent = self.encode_video(input_image_tensor, **tiler_kwargs)[0]  # (C, 1, H/8, W/8)
        
        # Latent shape
        latent_C = start_latent.shape[0]  # 16
        latent_H = start_latent.shape[2]  # H // 8
        latent_W = start_latent.shape[3]  # W // 8

        if (
            memory_policy
            in {
                "rarity_irreplaceability",
                "slam_covisibility",
                "slam_max_coverage",
                "slam_ri_blend",
                "facility_coreset",
                "kcenter_coreset",
                "trajectory_coverage",
                "density_balanced_view_coverage",
                "future_view_coverage",
                "mce",
                "h2o_heavy_hitter",
                "surprise_forcing",
            }
            and memory_budget is not None
            and memory_budget < 2
        ):
            raise ValueError(f"{memory_policy} requires memory_budget >= 2")
        if memory_bank_device not in {"cpu", "cuda"}:
            raise ValueError("memory_bank_device must be either 'cpu' or 'cuda'")
        if memory_bank_device == "cuda" and torch.device(self.device).type != "cuda":
            raise ValueError("memory_bank_device='cuda' requires a CUDA pipeline device")
        if int(ri_rarity_neighbors) < 1:
            raise ValueError("ri_rarity_neighbors must be at least 1")
        if int(slamri_rarity_neighbors) < 1:
            raise ValueError("slamri_rarity_neighbors must be at least 1")
        bank_device = torch.device(self.device if memory_bank_device == "cuda" else "cpu")
        
        # ============ 存储结构 ============
        all_section_latents = []  # (0:start_latent, else:1+19latents)
        all_generated_frames = {} # {frame_idx:frame_tensor}
        output_frame_sections = []
        pinned_memory_frames = (
            {0}
            if memory_policy
            in {
                "rarity_irreplaceability",
                "slam_covisibility",
                "slam_max_coverage",
                "slam_ri_blend",
                "trajectory_coverage",
                "density_balanced_view_coverage",
                "future_view_coverage",
                "mce",
                "surprise_forcing",
            }
            else set()
        )
        memory_buffer = FrameMemoryBuffer(
            policy=memory_policy,
            budget=memory_budget,
            pinned_frames=pinned_memory_frames,
        )
        visual_feature_extractor = None
        memory_dino_features = {}
        memory_rgb_features = {}
        memory_quality_scores = {}
        surprise_value_features = {}
        memory_coverage_masses = (
            {0: 1.0}
            if memory_policy in {"density_balanced_view_coverage", "future_view_coverage"}
            else {}
        )
        coreset_archive_frame_indices = []
        coreset_archive_seen = set()
        surprise_controller = None
        surprise_query_frame = None
        if memory_policy == "surprise_forcing":
            if not 0 <= int(surprise_value_layer) < len(self.dit.blocks):
                raise ValueError(
                    f"Invalid Surprise Forcing value layer: {surprise_value_layer}"
                )
            external_capacity = int(memory_budget) - len(pinned_memory_frames)
            if external_capacity < 1:
                raise ValueError(
                    "Surprise Forcing needs at least one external slot in addition "
                    "to the pinned sink frame"
                )
            surprise_controller = SurpriseForcingMemoryController(
                capacity=external_capacity,
                alpha=surprise_alpha,
                ema_momentum=surprise_ema_momentum,
                controller_step=surprise_controller_step,
                target_admission_ratio=surprise_target_admission_ratio,
                initial_threshold=surprise_initial_threshold,
                surprise_weight=surprise_surprise_weight,
                usage_weight=surprise_usage_weight,
                age_weight=surprise_age_weight,
                warmup_sections=surprise_warmup_sections,
            )
        if memory_policy in VISUAL_MEMORY_POLICIES:
            self.load_models_to_device([])
            visual_feature_extractor = VisualMemoryFeatureExtractor(device=self.device)
        section_start_frames = [i * (FRAMES_PER_SECTION - 1) for i in range(total_sections)]
        
        # 初始化: section 0 的 anchor 来自输入图片
        all_section_latents.append(start_latent)  # (C, 1, H, W) 作为 section -1 的 "latent"
        all_generated_frames[0] = input_image_tensor.detach().to(bank_device).clone()
        del input_image_tensor
        memory_buffer.add(0)
        print(
            f"Memory policy: {memory_policy}, budget: {memory_budget}, "
            f"bank device: {memory_bank_device}, stored frames: {len(memory_buffer)}"
        )
        access_trace_handle = None
        access_trace_metadata = dict(access_trace_metadata or {})
        if access_trace_path is not None:
            os.makedirs(os.path.dirname(access_trace_path) or ".", exist_ok=True)
            access_trace_handle = open(access_trace_path, "w", encoding="utf-8")

        attention_audit_handle = None
        attention_audit_event_count = 0
        attention_audit_metadata = dict(attention_audit_metadata or {})
        if attention_audit_enabled:
            os.makedirs(os.path.dirname(attention_audit_path) or ".", exist_ok=True)
            attention_audit_handle = open(
                attention_audit_path,
                "w",
                encoding="utf-8",
            )

        profile_metadata = {
            **dict(access_trace_metadata or {}),
            **dict(profile_metadata or {}),
            "memory_policy": memory_policy,
            "memory_budget": memory_budget,
            "memory_bank_device": memory_bank_device,
            "surprise_alpha": float(surprise_alpha),
            "surprise_ema_momentum": float(surprise_ema_momentum),
            "surprise_controller_step": float(surprise_controller_step),
            "surprise_target_admission_ratio": float(
                surprise_target_admission_ratio
            ),
            "surprise_initial_threshold": float(surprise_initial_threshold),
            "surprise_surprise_weight": float(surprise_surprise_weight),
            "surprise_usage_weight": float(surprise_usage_weight),
            "surprise_age_weight": float(surprise_age_weight),
            "surprise_route_top_k": int(surprise_route_top_k),
            "surprise_value_layer": int(surprise_value_layer),
            "surprise_warmup_sections": int(surprise_warmup_sections),
            "total_frames": int(total_frames),
            "total_sections": int(total_sections),
        }
        profiler = MemoryRolloutProfiler(
            path=profile_path,
            metadata=profile_metadata,
            cuda_device=self.device,
        )
        profiler.start_rollout(
            stored_memory_size=len(memory_buffer),
            bank_frame_bytes=tensor_mapping_nbytes(all_generated_frames),
        )

        def write_access_trace(payload):
            if access_trace_handle is None:
                return
            payload = {
                **access_trace_metadata,
                **payload,
            }
            dataset_start_frame = payload.get("dataset_start_frame")
            if dataset_start_frame is not None:
                if payload.get("target_frame") is not None:
                    payload["target_dataset_frame"] = int(dataset_start_frame) + int(payload["target_frame"])
                if payload.get("selected_memory_frame") is not None:
                    payload["selected_dataset_frame"] = int(dataset_start_frame) + int(payload["selected_memory_frame"])
                if payload.get("evicted_memory_frame") is not None:
                    payload["evicted_dataset_frame"] = int(dataset_start_frame) + int(payload["evicted_memory_frame"])
                if payload.get("candidate_memory_frame") is not None:
                    payload["candidate_dataset_frame"] = int(dataset_start_frame) + int(payload["candidate_memory_frame"])
                if payload.get("query_memory_frame") is not None:
                    payload["query_dataset_frame"] = int(dataset_start_frame) + int(payload["query_memory_frame"])
            access_trace_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        def write_attention_audit(payload):
            nonlocal attention_audit_event_count
            if attention_audit_handle is None:
                return
            payload = {
                **attention_audit_metadata,
                **payload,
            }
            dataset_start_frame = payload.get("dataset_start_frame")
            if dataset_start_frame is not None and payload.get("memory_frame") is not None:
                payload["memory_dataset_frame"] = (
                    int(dataset_start_frame) + int(payload["memory_frame"])
                )
            attention_audit_handle.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            attention_audit_handle.flush()
            attention_audit_event_count += 1

        def set_attention_collector(collector):
            for layer_idx in probe_layers:
                self.dit.blocks[layer_idx].self_attn.memory_attention_probe = (
                    collector.callback(layer_idx)
                )

        def clear_attention_collector():
            for layer_idx in probe_layers:
                self.dit.blocks[layer_idx].self_attn.memory_attention_probe = None

        def clear_surprise_value_collector():
            if memory_policy == "surprise_forcing":
                self.dit.blocks[
                    int(surprise_value_layer)
                ].self_attn.memory_value_probe = None

        clear_attention_collector()
        clear_surprise_value_collector()
        
        # Vanilla Sampling
        for section_idx in range(generated_section_count):
            profiler.begin_section(section_idx)
            print(f"Generating section {section_idx + 1}/{generated_section_count}")
            section_start_frame = section_start_frames[section_idx]
            section_candidate_count = 0
            section_host_to_device_bytes = 0
            
            # ============ 获取anchor latent + 确定帧范围 ============
            if section_idx == 0:
                # Section 0: anchor = start_latent (1f1l, 用户输入图片)
                anchor_latent = all_section_latents[0]  # (C, 1, H, W) - start_latent
                anchor_latent = anchor_latent.to(dtype=self.torch_dtype, device=self.device)

                # anchor 的 pose frame: 当前clip的起始帧
                anchor_pose_frame = section_start_frame  # frame 0 for clip_idx=0
                
                # anchor 覆盖的帧范围 (单帧)
                anchor_frame_range = [section_start_frame]
                
                # predict 帧索引: frames 1, 5, 9, ..., 73
                predict_latent_frames = [section_start_frame + (i * 4 + 1) for i in range(TARGET_LENGTH - ANCHOR_LENGTH)]
                
                # predict 覆盖的帧范围 (frames 1-76)
                predict_frame_range = list(range(section_start_frame + 1, section_start_frame + FRAMES_PER_SECTION))
            else:
                # Section > 0: anchor = 上一个section的最后一个latent (4f1l)
                prev_section_latent = all_section_latents[section_idx]  # (C, 20, H, W)
                anchor_latent = prev_section_latent[:, -1:, :, :]  # (C, 1, H, W) - 4f1l
                anchor_latent = anchor_latent.to(dtype=self.torch_dtype, device=self.device)

                # anchor 的 pose frame: 用前一个clip最后一个latent的pose帧索引
                anchor_pose_frame = section_start_frame - 3  # frame 73
                
                # anchor 覆盖的帧范围 (前一个section的最后4帧: 73-76)
                anchor_frame_range = list(range(section_start_frame - 3, section_start_frame + 1))
                
                # predict 帧索引: frames 77, 81, 85, ..., 149
                predict_latent_frames = [section_start_frame + 1 + (i * 4) for i in range(TARGET_LENGTH - ANCHOR_LENGTH)]
                
                # predict 覆盖的帧范围 (frames 77-152)
                predict_frame_range = list(range(section_start_frame + 1, section_start_frame + FRAMES_PER_SECTION))
            
            # ============ 构建 Context ============
            context_latent_list = []
            context_frame_indices = []
            exclude_frames = set(anchor_frame_range) | set(predict_frame_range)
            context_target_frames = [section_start_frame + 1 + i for i in range(PREDICT_FRAMES)]
            selected_contexts = []
            selection_sources = {}

            if section_idx == 0:
                with profiler.phase("context_encode"):
                    for slot_idx, frame_idx in enumerate(context_target_frames):
                        context_latent_list.append(
                            torch.zeros(
                                latent_C,
                                1,
                                latent_H,
                                latent_W,
                                dtype=anchor_latent.dtype,
                                device=anchor_latent.device,
                            )
                        )
                        context_frame_indices.append(anchor_pose_frame)
                        write_access_trace(
                            {
                                "event": "context_access",
                                "selected": False,
                                "fallback_reason": "initial_section",
                                "section_idx": section_idx,
                                "context_slot": slot_idx,
                                "target_frame": int(frame_idx),
                                "anchor_pose_frame": int(anchor_pose_frame),
                                "candidate_count": 0,
                                "stored_memory_size": len(memory_buffer),
                                "memory_policy": memory_policy,
                                "memory_budget": memory_budget,
                                "memory_bank_device": memory_bank_device,
                            }
                        )
            else:
                candidate_frame_indices = memory_buffer.candidates(exclude_frames=exclude_frames)
                if memory_policy == "surprise_forcing":
                    query_descriptor = surprise_value_features.get(anchor_pose_frame)
                    if query_descriptor is None:
                        raise RuntimeError(
                            "Missing Surprise Forcing query descriptor for anchor "
                            f"frame {anchor_pose_frame}"
                        )
                    surprise_bank_frames = set(surprise_controller.frames())
                    external_candidates = [
                        frame_idx
                        for frame_idx in candidate_frame_indices
                        if frame_idx in surprise_bank_frames
                    ]
                    routed_frames, route_similarities = surprise_controller.route(
                        query_descriptor=query_descriptor,
                        candidate_frames=external_candidates,
                        top_k=surprise_route_top_k,
                        record_usage=False,
                    )
                    sink_frames = [
                        frame_idx
                        for frame_idx in candidate_frame_indices
                        if frame_idx in pinned_memory_frames
                    ]
                    candidate_frame_indices = list(
                        dict.fromkeys(sink_frames + routed_frames)
                    )
                    write_access_trace(
                        {
                            "event": "surprise_routing",
                            "section_idx": int(section_idx),
                            "anchor_pose_frame": int(anchor_pose_frame),
                            "query_memory_frame": int(anchor_pose_frame),
                            "bank_candidate_count": len(external_candidates),
                            "route_top_k": int(surprise_route_top_k),
                            "routed_memory_frames": [
                                int(frame_idx) for frame_idx in routed_frames
                            ],
                            "sink_memory_frames": [
                                int(frame_idx) for frame_idx in sink_frames
                            ],
                            "routed_similarities": {
                                str(frame_idx): float(route_similarities[frame_idx])
                                for frame_idx in routed_frames
                            },
                            "memory_policy": memory_policy,
                            "memory_budget": memory_budget,
                            "stored_memory_size": len(memory_buffer),
                        }
                    )
                section_candidate_count = len(candidate_frame_indices)
                print(f"  Selecting context frames (1 per target, {PREDICT_FRAMES} targets)...")
                print(
                    f"  Excluding frames: anchor={anchor_frame_range}, "
                    f"predict={predict_frame_range[0]}-{predict_frame_range[-1]}"
                )
                print(
                    f"  Candidate memory frames: {section_candidate_count} / "
                    f"stored={len(memory_buffer)}"
                )

                with profiler.phase("context_selection"):
                    for slot_idx, frame_idx in enumerate(context_target_frames):
                        target_c2w = c2ws[frame_idx]
                        override = context_selection_overrides.get(
                            (int(section_idx), int(frame_idx))
                        )
                        if override is not None:
                            best_idx = int(override["memory_frame"])
                            if best_idx not in candidate_frame_indices:
                                raise ValueError(
                                    "Replay override selected a frame outside the "
                                    f"current bank: section={section_idx}, "
                                    f"target={frame_idx}, memory={best_idx}"
                                )
                            best_iou = calculate_overlap_from_c2w(
                                target_c2w,
                                c2ws[best_idx],
                                fov_half_h=FOV_HALF_H,
                                fov_half_v=FOV_HALF_V,
                                num_samples=FOV_SAMPLES,
                                radius=FOV_RADIUS,
                                return_details=False,
                            )
                            selection_sources[(slot_idx, frame_idx)] = {
                                "selection_source": "override",
                                "override_source_run": override.get("source_run"),
                            }
                        else:
                            best_idx = None
                            best_iou = -1
                            for candidate_idx in candidate_frame_indices:
                                candidate_c2w = c2ws[candidate_idx]
                                iou = calculate_overlap_from_c2w(
                                    target_c2w,
                                    candidate_c2w,
                                    fov_half_h=FOV_HALF_H,
                                    fov_half_v=FOV_HALF_V,
                                    num_samples=FOV_SAMPLES,
                                    radius=FOV_RADIUS,
                                    return_details=False,
                                )
                                if iou > best_iou:
                                    best_iou = iou
                                    best_idx = candidate_idx
                            selection_sources[(slot_idx, frame_idx)] = {
                                "selection_source": "retriever",
                                "override_source_run": None,
                            }
                        selected_contexts.append((slot_idx, frame_idx, best_idx, best_iou))

                if memory_policy == "surprise_forcing":
                    selected_surprise_frames = sorted(
                        {
                            int(best_idx)
                            for _, _, best_idx, _ in selected_contexts
                            if best_idx in surprise_bank_frames
                        }
                    )
                    surprise_controller.record_usage(selected_surprise_frames)
                    write_access_trace(
                        {
                            "event": "surprise_usage",
                            "section_idx": int(section_idx),
                            "actually_retrieved_memory_frames": (
                                selected_surprise_frames
                            ),
                            "actually_retrieved_count": len(
                                selected_surprise_frames
                            ),
                            "memory_policy": memory_policy,
                            "memory_budget": memory_budget,
                        }
                    )

                with profiler.phase("context_encode"):
                    self.load_models_to_device(["vae"])
                    for slot_idx, frame_idx, best_idx, best_iou in selected_contexts:
                        if best_idx is not None and best_idx in all_generated_frames:
                            memory_buffer.record_selection(best_idx, best_iou)
                            chosen_frame = all_generated_frames[best_idx]
                            if chosen_frame.device.type == "cpu":
                                section_host_to_device_bytes += (
                                    chosen_frame.numel() * chosen_frame.element_size()
                                )
                            chosen_frame = chosen_frame.to(
                                dtype=self.torch_dtype,
                                device=self.device,
                            )
                            chosen_latent = self.encode_video(chosen_frame, **tiler_kwargs)[0]
                            context_latent_list.append(chosen_latent)
                            context_frame_indices.append(best_idx)
                            write_access_trace(
                                {
                                    "event": "context_access",
                                    "selected": True,
                                    "fallback_reason": None,
                                    "section_idx": section_idx,
                                    "context_slot": slot_idx,
                                    "target_frame": int(frame_idx),
                                    "anchor_pose_frame": int(anchor_pose_frame),
                                    "selected_memory_frame": int(best_idx),
                                    "memory_age": int(frame_idx - best_idx),
                                    "selected_overlap": float(best_iou),
                                    "selected_count_after": memory_buffer.selected_count(best_idx),
                                    "candidate_count": section_candidate_count,
                                    "candidate_min_frame": (
                                        int(min(candidate_frame_indices))
                                        if candidate_frame_indices
                                        else None
                                    ),
                                    "candidate_max_frame": (
                                        int(max(candidate_frame_indices))
                                        if candidate_frame_indices
                                        else None
                                    ),
                                    "stored_memory_size": len(memory_buffer),
                                    "memory_policy": memory_policy,
                                    "memory_budget": memory_budget,
                                    "memory_bank_device": memory_bank_device,
                                    **selection_sources.get(
                                        (slot_idx, frame_idx),
                                        {
                                            "selection_source": "retriever",
                                            "override_source_run": None,
                                        },
                                    ),
                                }
                            )
                        else:
                            context_latent_list.append(
                                torch.zeros(
                                    latent_C,
                                    1,
                                    latent_H,
                                    latent_W,
                                    dtype=anchor_latent.dtype,
                                    device=anchor_latent.device,
                                )
                            )
                            context_frame_indices.append(anchor_pose_frame)
                            write_access_trace(
                                {
                                    "event": "context_access",
                                    "selected": False,
                                    "fallback_reason": "no_valid_context",
                                    "section_idx": section_idx,
                                    "context_slot": slot_idx,
                                    "target_frame": int(frame_idx),
                                    "anchor_pose_frame": int(anchor_pose_frame),
                                    "candidate_count": section_candidate_count,
                                    "stored_memory_size": len(memory_buffer),
                                    "memory_policy": memory_policy,
                                    "memory_budget": memory_budget,
                                    "memory_bank_device": memory_bank_device,
                                }
                            )

                print(f"  Selected [{context_frame_indices}] as context frames")
            
            # 拼接context: (C, context_length, H, W)
            context_latent = torch.cat(context_latent_list, dim=1)
            
            # Context pose: 相对于anchor
            context_pose = compute_relative_pose(c2ws, anchor_pose_frame, context_frame_indices)  # (context_length, 12)
            
            # ============ 准备 Target (1个anchor + 19帧噪声 = 20帧) ============
            # 生成19帧噪声
            noise_latents = self.generate_noise((1, latent_C, TARGET_LENGTH - ANCHOR_LENGTH, latent_H, latent_W), seed=seed, device=rand_device, dtype=torch.float32).to(dtype=self.torch_dtype, device=self.device)

            # Target pose: 20帧 (1 anchor + 19predict)
            target_latent_frames = [anchor_pose_frame] + predict_latent_frames
            target_pose = compute_relative_pose(c2ws, anchor_pose_frame, target_latent_frames)  # (20, 12)
            
            # ============ Denoising ============
            with profiler.phase("denoising"):
                self.load_models_to_device(["dit"])
                context_latent_input = context_latent.unsqueeze(0).to(
                    dtype=self.torch_dtype,
                    device=self.device,
                )
                context_pose_input = context_pose.unsqueeze(0).to(
                    dtype=self.torch_dtype,
                    device=self.device,
                )
                target_pose_input = target_pose.unsqueeze(0).to(
                    dtype=self.torch_dtype,
                    device=self.device,
                )
                anchor_latent_batch = anchor_latent.unsqueeze(0).to(
                    dtype=self.torch_dtype,
                    device=self.device,
                )

                context_spatial = (
                    ((latent_H + 3) // 4) * ((latent_W + 3) // 4)
                )
                target_spatial = (latent_H // 2) * (latent_W // 2)
                surprise_value_collector = None

                for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                    timestep = timestep.unsqueeze(0).to(
                        dtype=self.torch_dtype,
                        device=self.device,
                    )
                    target_input = torch.cat([anchor_latent_batch, noise_latents], dim=2)
                    run_attention_probe = (
                        attention_audit_enabled
                        and section_idx in probe_sections
                        and progress_id in probe_steps
                    )
                    attention_collector = None
                    if run_attention_probe:
                        attention_collector = MemoryAttentionCollector(
                            context_spatial=context_spatial,
                            context_frame_indices=context_frame_indices,
                            query_count=attention_query_count,
                            query_chunk_size=attention_query_chunk_size,
                        )
                        set_attention_collector(attention_collector)
                    capture_surprise_values = (
                        memory_policy == "surprise_forcing"
                        and progress_id == num_inference_steps - 1
                    )
                    if capture_surprise_values:
                        surprise_value_collector = TargetValueDescriptorCollector(
                            context_token_count=PREDICT_FRAMES * context_spatial,
                            target_length=TARGET_LENGTH,
                            target_spatial=target_spatial,
                        )
                        self.dit.blocks[
                            int(surprise_value_layer)
                        ].self_attn.memory_value_probe = (
                            surprise_value_collector.capture
                        )
                    try:
                        noise_pred_posi = self.forward(
                            context_latents=context_latent_input,
                            target_latents=target_input,
                            context_pose=context_pose_input,
                            target_pose=target_pose_input,
                            timestep=timestep,
                            context=prompt_emb_posi["context"],
                        )
                    finally:
                        clear_attention_collector()
                        clear_surprise_value_collector()

                    if cfg_scale != 1.0:
                        noise_pred_nega = self.forward(
                            context_latents=context_latent_input,
                            target_latents=target_input,
                            context_pose=context_pose_input,
                            target_pose=target_pose_input,
                            timestep=timestep,
                            context=prompt_emb_nega["context"],
                        )
                        noise_pred = noise_pred_nega + cfg_scale * (
                            noise_pred_posi - noise_pred_nega
                        )
                    else:
                        noise_pred = noise_pred_posi

                    if run_attention_probe:
                        attention_summary = attention_collector.aggregate()
                        memory_scores = add_retrieval_controls(
                            attention_summary["memory_scores"],
                            selected_contexts,
                            section_start_frame,
                        )
                        for row in memory_scores:
                            dataset_start_frame = attention_audit_metadata.get(
                                "dataset_start_frame"
                            )
                            if dataset_start_frame is not None:
                                row["memory_dataset_frame"] = (
                                    int(dataset_start_frame)
                                    + int(row["memory_frame"])
                                )

                        write_attention_audit(
                            {
                                "event": "attention_probe",
                                "section_idx": int(section_idx),
                                "section_start_frame": int(section_start_frame),
                                "progress_id": int(progress_id),
                                "timestep": float(timestep.float().item()),
                                "probe_layers": attention_summary["probe_layers"],
                                "query_count": attention_summary["query_count"],
                                "context_token_count": attention_summary[
                                    "context_token_count"
                                ],
                                "target_token_count": attention_summary[
                                    "target_token_count"
                                ],
                                "context_attention_mass": attention_summary[
                                    "context_attention_mass"
                                ],
                                "unique_retrieved_memories": len(memory_scores),
                                "memory_scores": memory_scores,
                            }
                        )

                        intervention_seed = (
                            int(seed or 0) * 1_000_003
                            + int(section_idx) * 1_009
                            + int(progress_id)
                        )
                        candidates = select_intervention_candidates(
                            memory_scores,
                            intervention_seed,
                        )
                        baseline_prediction = noise_pred[
                            :, :, ANCHOR_LENGTH:, :, :
                        ].float()
                        baseline_energy = float(
                            baseline_prediction.square().mean().item()
                        )

                        for candidate in candidates:
                            slots = [
                                int(value) for value in candidate["context_slots"]
                            ]
                            ablated_context_latents = context_latent_input.clone()
                            ablated_context_pose = context_pose_input.clone()
                            ablated_context_latents[:, :, slots, :, :] = 0
                            ablated_context_pose[:, slots, :] = target_pose_input[
                                :, :1, :
                            ].expand(-1, len(slots), -1)

                            ablated_posi = self.forward(
                                context_latents=ablated_context_latents,
                                target_latents=target_input,
                                context_pose=ablated_context_pose,
                                target_pose=target_pose_input,
                                timestep=timestep,
                                context=prompt_emb_posi["context"],
                            )
                            if cfg_scale != 1.0:
                                ablated_nega = self.forward(
                                    context_latents=ablated_context_latents,
                                    target_latents=target_input,
                                    context_pose=ablated_context_pose,
                                    target_pose=target_pose_input,
                                    timestep=timestep,
                                    context=prompt_emb_nega["context"],
                                )
                                ablated_prediction = ablated_nega + cfg_scale * (
                                    ablated_posi - ablated_nega
                                )
                            else:
                                ablated_nega = None
                                ablated_prediction = ablated_posi

                            ablated_prediction = ablated_prediction[
                                :, :, ANCHOR_LENGTH:, :, :
                            ].float()
                            prediction_delta = (
                                ablated_prediction - baseline_prediction
                            )
                            delta_mse = float(
                                prediction_delta.square().mean().item()
                            )
                            relative_l2 = math.sqrt(
                                delta_mse / max(baseline_energy, 1e-12)
                            )
                            prediction_cosine = float(
                                F.cosine_similarity(
                                    baseline_prediction.flatten(),
                                    ablated_prediction.flatten(),
                                    dim=0,
                                ).item()
                            )

                            write_attention_audit(
                                {
                                    "event": "memory_intervention",
                                    "section_idx": int(section_idx),
                                    "section_start_frame": int(section_start_frame),
                                    "progress_id": int(progress_id),
                                    "timestep": float(timestep.float().item()),
                                    "probe_layers": attention_summary[
                                        "probe_layers"
                                    ],
                                    "query_count": attention_summary[
                                        "query_count"
                                    ],
                                    "context_attention_mass": attention_summary[
                                        "context_attention_mass"
                                    ],
                                    "intervention": "null_all_retrieval_slots",
                                    "intervention_role": candidate[
                                        "intervention_role"
                                    ],
                                    "memory_frame": int(
                                        candidate["memory_frame"]
                                    ),
                                    "context_slots": slots,
                                    "slot_count": int(candidate["slot_count"]),
                                    "attention_total": float(
                                        candidate["attention_total"]
                                    ),
                                    "attention_per_slot": float(
                                        candidate["attention_per_slot"]
                                    ),
                                    "retrieval_overlap_mean": candidate[
                                        "retrieval_overlap_mean"
                                    ],
                                    "retrieval_overlap_max": candidate[
                                        "retrieval_overlap_max"
                                    ],
                                    "memory_age_mean": float(
                                        candidate["memory_age_mean"]
                                    ),
                                    "prediction_delta_mse": delta_mse,
                                    "prediction_relative_l2": relative_l2,
                                    "prediction_cosine": prediction_cosine,
                                    "baseline_prediction_energy": baseline_energy,
                                    "ablated_prediction_energy": float(
                                        ablated_prediction.square().mean().item()
                                    ),
                                }
                            )
                            del (
                                ablated_context_latents,
                                ablated_context_pose,
                                ablated_posi,
                                ablated_prediction,
                                prediction_delta,
                            )
                            if ablated_nega is not None:
                                del ablated_nega

                    noise_pred_rest = noise_pred[:, :, ANCHOR_LENGTH:, :, :]
                    noise_latents = self.scheduler.step(
                        noise_pred_rest,
                        self.scheduler.timesteps[progress_id],
                        noise_latents,
                    )

                if memory_policy == "surprise_forcing":
                    if (
                        surprise_value_collector is None
                        or surprise_value_collector.descriptors is None
                    ):
                        raise RuntimeError(
                            "Surprise Forcing failed to capture final-step value descriptors"
                        )
                    if len(target_latent_frames) != len(
                        surprise_value_collector.descriptors
                    ):
                        raise RuntimeError(
                            "Surprise Forcing descriptor count does not match target frames"
                        )
                    for descriptor_idx, frame_idx in enumerate(target_latent_frames):
                        surprise_value_features[int(frame_idx)] = (
                            np.array(
                                surprise_value_collector.descriptors[descriptor_idx],
                                dtype=np.float32,
                                copy=True,
                            )
                        )
                    surprise_query_frame = int(predict_latent_frames[-1])

            # Decode once. The same CPU section is retained for final output, avoiding
            # a second full-video decode and its unrelated VRAM peak.
            decode_phase_start = profiler.start_phase()
            section_start_source = all_generated_frames[section_start_frame]
            if section_start_source.device.type == "cpu":
                section_host_to_device_bytes += (
                    section_start_source.numel() * section_start_source.element_size()
                )
            section_start_latent = self.encode_video(
                section_start_source.to(dtype=self.torch_dtype, device=self.device),
                **tiler_kwargs,
            )[0]
            if (
                memory_policy == "surprise_forcing"
                and section_start_frame not in set(memory_buffer.candidates())
            ):
                all_generated_frames.pop(section_start_frame, None)
            section_full_latent = torch.cat(
                [section_start_latent, noise_latents.squeeze(0)],
                dim=1,
            )
            all_section_latents.append(section_full_latent.cpu())

            self.load_models_to_device(["vae"])
            section_frames = self.decode_video(
                section_full_latent.unsqueeze(0).to(
                    dtype=self.torch_dtype,
                    device=self.device,
                ),
                **tiler_kwargs,
            )
            section_frames_cpu = section_frames.detach().cpu()
            output_section = section_frames_cpu.squeeze(0)
            if section_idx > 0:
                output_section = output_section[:, 1:, :, :]
            output_frame_sections.append(output_section)
            profiler.end_phase("section_decode", decode_phase_start)

            new_frame_indices = range(
                section_start_frame,
                section_start_frame + section_frames_cpu.shape[2],
            )
            eviction_scores = None
            eviction_score_details = {}
            section_end_frame = section_start_frame + section_frames_cpu.shape[2] - 1
            protected_frames = {section_end_frame}
            policy_phase_start = profiler.start_phase()
            if memory_policy in VISUAL_MEMORY_POLICIES:
                feature_frame_indices = list(new_frame_indices)
                feature_images = [
                    self.frame_tensor_to_pil(
                        section_frames_cpu[
                            :,
                            :,
                            frame_idx - section_start_frame : frame_idx - section_start_frame + 1,
                            :,
                            :,
                        ]
                    )
                    for frame_idx in feature_frame_indices
                ]
                dino_batch, rgb_batch = visual_feature_extractor.encode_pil_images(feature_images)
                quality_batch = image_quality_scores_from_pil_images(feature_images)
                for feature_idx, frame_idx in enumerate(feature_frame_indices):
                    memory_dino_features[frame_idx] = dino_batch[feature_idx]
                    memory_rgb_features[frame_idx] = rgb_batch[feature_idx]
                    if memory_policy not in {
                        "density_balanced_view_coverage",
                        "future_view_coverage",
                        "mce",
                        "slam_max_coverage",
                    }:
                        memory_quality_scores[frame_idx] = float(quality_batch[feature_idx])

            if memory_policy in ARCHIVE_MEMORY_POLICIES:
                for frame_idx in new_frame_indices:
                    if frame_idx in coreset_archive_seen:
                        continue
                    if frame_idx == 0 or frame_idx % CORESET_ARCHIVE_STRIDE == 0:
                        coreset_archive_frame_indices.append(frame_idx)
                        coreset_archive_seen.add(frame_idx)
                if len(coreset_archive_frame_indices) > CORESET_MAX_ARCHIVE_SIZE:
                    keep_positions = np.linspace(
                        0,
                        len(coreset_archive_frame_indices) - 1,
                        CORESET_MAX_ARCHIVE_SIZE,
                        dtype=np.int64,
                    )
                    coreset_archive_frame_indices = [
                        coreset_archive_frame_indices[int(position)]
                        for position in keep_positions
                    ]
                    coreset_archive_seen = set(coreset_archive_frame_indices)

            if memory_policy == "rarity_irreplaceability":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                eviction_scores, eviction_score_details = compute_rarity_irreplaceability_scores(
                    memory_frame_indices=prospective_memory,
                    pinned_frames=pinned_memory_frames,
                    rarity_neighbors=ri_rarity_neighbors,
                    dino_features=memory_dino_features,
                    rgb_features=memory_rgb_features,
                    return_details=True,
                )
            elif memory_policy == "slam_covisibility":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                eviction_scores, eviction_score_details = compute_slam_covisibility_scores(
                    memory_frame_indices=prospective_memory,
                    c2ws=c2ws,
                    pinned_frames=pinned_memory_frames,
                    dino_features=memory_dino_features,
                    rgb_features=memory_rgb_features,
                    return_details=True,
                )
            elif memory_policy == "slam_ri_blend":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                eviction_scores, eviction_score_details = compute_slam_ri_blend_scores(
                    memory_frame_indices=prospective_memory,
                    c2ws=c2ws,
                    forced_keep_frames=pinned_memory_frames,
                    dino_features=memory_dino_features,
                    rgb_features=memory_rgb_features,
                    beta=slamri_beta,
                    ri_kwargs={"rarity_neighbors": slamri_rarity_neighbors},
                    return_details=True,
                )
            elif memory_policy == "slam_max_coverage":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                eviction_scores, eviction_score_details = (
                    compute_slam_max_coverage_scores(
                        memory_frame_indices=prospective_memory,
                        c2ws=c2ws,
                        budget=memory_budget,
                        forced_keep_frames=(
                            protected_frames | pinned_memory_frames
                        ),
                        dino_features=memory_dino_features,
                        rgb_features=memory_rgb_features,
                        return_details=True,
                    )
                )
            elif memory_policy == "facility_coreset":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                eviction_scores, eviction_score_details = compute_facility_coreset_scores(
                    memory_frame_indices=prospective_memory,
                    archive_frame_indices=coreset_archive_frame_indices,
                    c2ws=c2ws,
                    budget=memory_budget,
                    forced_keep_frames=protected_frames,
                    dino_features=memory_dino_features,
                    frame_quality=memory_quality_scores,
                    return_details=True,
                )
            elif memory_policy == "kcenter_coreset":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                eviction_scores, eviction_score_details = compute_kcenter_coreset_scores(
                    memory_frame_indices=prospective_memory,
                    archive_frame_indices=coreset_archive_frame_indices,
                    c2ws=c2ws,
                    budget=memory_budget,
                    forced_keep_frames=protected_frames,
                    dino_features=memory_dino_features,
                    return_details=True,
                )
            elif memory_policy == "trajectory_coverage":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                eviction_scores, eviction_score_details = compute_trajectory_coverage_scores(
                    memory_frame_indices=prospective_memory,
                    archive_frame_indices=coreset_archive_frame_indices,
                    c2ws=c2ws,
                    budget=memory_budget,
                    forced_keep_frames=protected_frames | pinned_memory_frames,
                    fov_half_h=FOV_HALF_H,
                    fov_half_v=FOV_HALF_V,
                    radius=FOV_RADIUS,
                    return_details=True,
                )
            elif memory_policy == "density_balanced_view_coverage":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                prospective_masses = {
                    frame_idx: memory_coverage_masses.get(frame_idx, 1.0)
                    for frame_idx in prospective_memory
                }
                eviction_scores, eviction_score_details = (
                    compute_density_balanced_view_coverage_scores(
                        memory_frame_indices=prospective_memory,
                        c2ws=c2ws,
                        budget=memory_budget,
                        forced_keep_frames=protected_frames | pinned_memory_frames,
                        dino_features=memory_dino_features,
                        rgb_features=memory_rgb_features,
                        coverage_masses=prospective_masses,
                        density_alpha=density_coverage_alpha,
                        dino_weight=density_coverage_dino_weight,
                        rgb_weight=density_coverage_rgb_weight,
                        fov_half_h=FOV_HALF_H,
                        fov_half_v=FOV_HALF_V,
                        radius=FOV_RADIUS,
                        return_details=True,
                    )
                )
            elif memory_policy == "future_view_coverage":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                prospective_masses = {
                    frame_idx: memory_coverage_masses.get(frame_idx, 1.0)
                    for frame_idx in prospective_memory
                }
                future_query_frame_indices = range(
                    section_end_frame + 1,
                    total_frames,
                    max(1, int(future_coverage_query_stride)),
                )
                eviction_scores, eviction_score_details = (
                    compute_future_view_coverage_scores(
                        memory_frame_indices=prospective_memory,
                        c2ws=c2ws,
                        budget=memory_budget,
                        future_query_frame_indices=future_query_frame_indices,
                        forced_keep_frames=protected_frames | pinned_memory_frames,
                        dino_features=memory_dino_features,
                        rgb_features=memory_rgb_features,
                        coverage_masses=prospective_masses,
                        density_alpha=future_coverage_alpha,
                        dino_weight=future_coverage_dino_weight,
                        rgb_weight=future_coverage_rgb_weight,
                        future_query_weight=future_coverage_query_weight,
                        fov_half_h=FOV_HALF_H,
                        fov_half_v=FOV_HALF_V,
                        radius=FOV_RADIUS,
                        return_details=True,
                    )
                )
            elif memory_policy == "mce":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                future_query_frame_indices = range(
                    section_end_frame + 1,
                    total_frames,
                    max(1, int(mce_query_stride)),
                )
                eviction_scores, eviction_score_details = (
                    compute_marginal_coverage_eviction_scores(
                        memory_frame_indices=prospective_memory,
                        c2ws=c2ws,
                        budget=memory_budget,
                        future_query_frame_indices=future_query_frame_indices,
                        forced_keep_frames=protected_frames | pinned_memory_frames,
                        dino_features=memory_dino_features,
                        rgb_features=memory_rgb_features,
                        alpha=mce_alpha,
                        lambda_hist=mce_lambda,
                        gamma=mce_gamma,
                        rarity_neighbors=mce_rarity_neighbors,
                        fov_half_h=FOV_HALF_H,
                        fov_half_v=FOV_HALF_V,
                        radius=FOV_RADIUS,
                        return_details=True,
                    )
                )
            elif memory_policy == "surprise_forcing":
                evicted_frames = []
                for frame_idx in predict_latent_frames:
                    descriptor = surprise_value_features.get(int(frame_idx))
                    if descriptor is None:
                        raise RuntimeError(
                            "Missing Surprise Forcing value descriptor for frame "
                            f"{frame_idx}"
                        )
                    decision = surprise_controller.consider(
                        frame_idx=int(frame_idx),
                        descriptor=descriptor,
                        section_idx=int(section_idx),
                        current_frame=int(frame_idx),
                    )
                    replaced_frame = decision.get("evicted_frame")
                    if replaced_frame is not None:
                        memory_buffer.remove(int(replaced_frame))
                        evicted_frames.append(int(replaced_frame))
                        eviction_score_details[int(replaced_frame)] = {
                            "score": decision.get("minimum_bank_priority"),
                            "surprise_priority": decision.get(
                                "minimum_bank_priority"
                            ),
                            "surprise_replaced_by": int(frame_idx),
                        }
                    if decision["committed"]:
                        memory_buffer.add(int(frame_idx), evict=False)

                    write_access_trace(
                        {
                            "event": "surprise_admission",
                            "section_idx": int(section_idx),
                            "candidate_memory_frame": int(frame_idx),
                            "section_end_frame": int(section_end_frame),
                            **{
                                key: value
                                for key, value in decision.items()
                                if key not in {"frame_idx", "section_idx"}
                            },
                            "bank_size_after": len(surprise_controller.frames()),
                            "stored_memory_size_after": len(memory_buffer),
                            "memory_policy": memory_policy,
                            "memory_budget": memory_budget,
                            "surprise_alpha": float(surprise_alpha),
                            "surprise_ema_momentum": float(
                                surprise_ema_momentum
                            ),
                            "surprise_controller_step": float(
                                surprise_controller_step
                            ),
                            "surprise_route_top_k": int(surprise_route_top_k),
                            "surprise_value_layer": int(surprise_value_layer),
                        }
                    )

                surprise_state = surprise_controller.state_snapshot(
                    current_frame=section_end_frame
                )
                memory_buffer.set_scores(
                    {
                        frame_idx: row["priority"]
                        for frame_idx, row in surprise_state.items()
                    }
                )
                write_access_trace(
                    {
                        "event": "surprise_bank_update",
                        "section_idx": int(section_idx),
                        "section_end_frame": int(section_end_frame),
                        "bank_frames": [
                            int(frame_idx) for frame_idx in surprise_controller.frames()
                        ],
                        "bank_state": {
                            str(frame_idx): row
                            for frame_idx, row in surprise_state.items()
                        },
                        "sink_frames": sorted(
                            int(frame_idx) for frame_idx in pinned_memory_frames
                        ),
                        "ema_mean": float(surprise_controller.ema_mean),
                        "ema_variance": float(
                            surprise_controller.ema_variance
                        ),
                        "threshold": float(surprise_controller.threshold),
                        "evaluated_count": int(
                            surprise_controller.evaluated_count
                        ),
                        "gate_pass_count": int(
                            surprise_controller.gate_pass_count
                        ),
                        "commit_count": int(surprise_controller.commit_count),
                        "memory_policy": memory_policy,
                        "memory_budget": memory_budget,
                    }
                )

                keep_surprise_features = (
                    set(surprise_controller.frames())
                    | set(pinned_memory_frames)
                    | {int(surprise_query_frame)}
                )
                surprise_value_features = {
                    frame_idx: descriptor
                    for frame_idx, descriptor in surprise_value_features.items()
                    if frame_idx in keep_surprise_features
                }
            elif memory_policy == "h2o_heavy_hitter":
                current_memory = list(memory_buffer.candidates())
                prospective_memory = current_memory + [
                    frame_idx
                    for frame_idx in new_frame_indices
                    if frame_idx not in current_memory
                ]
                eviction_scores, eviction_score_details = compute_h2o_heavy_hitter_scores(
                    memory_frame_indices=prospective_memory,
                    memory_stats=memory_buffer.stats_snapshot(),
                    budget=memory_budget,
                    forced_keep_frames=protected_frames,
                    return_details=True,
                )

            if memory_policy != "surprise_forcing":
                evicted_frames = memory_buffer.update(
                    new_frame_indices,
                    eviction_scores=eviction_scores,
                    protected_frames=protected_frames,
                )
            if memory_policy == "density_balanced_view_coverage":
                retained_frames_after_update = memory_buffer.candidates()
                memory_coverage_masses = {
                    frame_idx: float(
                        eviction_score_details[frame_idx][
                            "density_coverage_assigned_mass"
                        ]
                    )
                    for frame_idx in retained_frames_after_update
                }
                representative = eviction_score_details[retained_frames_after_update[0]]
                write_access_trace(
                    {
                        "event": "density_coverage_update",
                        "section_idx": section_idx,
                        "section_end_frame": int(section_end_frame),
                        "stored_memory_size": len(memory_buffer),
                        "memory_policy": memory_policy,
                        "memory_budget": memory_budget,
                        "memory_bank_device": memory_bank_device,
                        "coverage_value": representative["density_coverage_value"],
                        "coverage_mean": representative["density_coverage_mean"],
                        "coverage_min": representative["density_coverage_min"],
                        "coverage_p10": representative["density_coverage_p10"],
                        "coverage_total_mass": representative[
                            "density_coverage_total_mass"
                        ],
                        "density_alpha": float(density_coverage_alpha),
                        "dino_weight": float(density_coverage_dino_weight),
                        "rgb_weight": float(density_coverage_rgb_weight),
                        "retained_memory_frames": [
                            int(frame_idx) for frame_idx in retained_frames_after_update
                        ],
                        "retained_coverage_masses": [
                            memory_coverage_masses[frame_idx]
                            for frame_idx in retained_frames_after_update
                        ],
                    }
                )
            elif memory_policy == "future_view_coverage":
                retained_frames_after_update = memory_buffer.candidates()
                memory_coverage_masses = {
                    frame_idx: float(
                        eviction_score_details[frame_idx][
                            "future_view_coverage_assigned_mass"
                        ]
                    )
                    for frame_idx in retained_frames_after_update
                }
                representative = eviction_score_details[retained_frames_after_update[0]]
                write_access_trace(
                    {
                        "event": "future_view_coverage_update",
                        "section_idx": section_idx,
                        "section_end_frame": int(section_end_frame),
                        "stored_memory_size": len(memory_buffer),
                        "memory_policy": memory_policy,
                        "memory_budget": memory_budget,
                        "memory_bank_device": memory_bank_device,
                        "coverage_value": representative["future_view_coverage_value"],
                        "future_coverage_value": representative[
                            "future_view_coverage_future_value"
                        ],
                        "future_query_count": representative[
                            "future_view_coverage_future_query_count"
                        ],
                        "coverage_total_mass": representative[
                            "future_view_coverage_total_mass"
                        ],
                        "density_alpha": float(future_coverage_alpha),
                        "dino_weight": float(future_coverage_dino_weight),
                        "rgb_weight": float(future_coverage_rgb_weight),
                        "future_query_weight": float(future_coverage_query_weight),
                        "future_query_stride": int(future_coverage_query_stride),
                        "retained_memory_frames": [
                            int(frame_idx) for frame_idx in retained_frames_after_update
                        ],
                        "retained_coverage_masses": [
                            memory_coverage_masses[frame_idx]
                            for frame_idx in retained_frames_after_update
                        ],
                    }
                )
            elif memory_policy == "mce":
                retained_frames_after_update = memory_buffer.candidates()
                representative = eviction_score_details[retained_frames_after_update[0]]
                write_access_trace(
                    {
                        "event": "mce_update",
                        "section_idx": section_idx,
                        "section_end_frame": int(section_end_frame),
                        "stored_memory_size": len(memory_buffer),
                        "memory_policy": memory_policy,
                        "memory_budget": memory_budget,
                        "memory_bank_device": memory_bank_device,
                        "coverage_value": representative["mce_coverage_value"],
                        "mce_alpha": float(mce_alpha),
                        "mce_lambda": representative["mce_lambda"],
                        "mce_gamma": float(mce_gamma),
                        "mce_query_stride": int(mce_query_stride),
                        "mce_rarity_neighbors": int(mce_rarity_neighbors),
                        "mce_num_hist_queries": representative["mce_num_hist_queries"],
                        "mce_num_ctrl_queries": representative["mce_num_ctrl_queries"],
                        "mce_hist_query_frames": representative["mce_hist_query_frames"],
                        "retained_memory_frames": [
                            int(frame_idx) for frame_idx in retained_frames_after_update
                        ],
                    }
                )
            elif memory_policy == "slam_max_coverage":
                retained_frames_after_update = memory_buffer.candidates()
                representative = eviction_score_details[
                    retained_frames_after_update[0]
                ]
                write_access_trace(
                    {
                        "event": "slam_max_coverage_update",
                        "section_idx": section_idx,
                        "section_end_frame": int(section_end_frame),
                        "stored_memory_size": len(memory_buffer),
                        "memory_policy": memory_policy,
                        "memory_budget": memory_budget,
                        "memory_bank_device": memory_bank_device,
                        "coverage_value": representative[
                            "slam_max_coverage_value"
                        ],
                        "coverage_mean": representative[
                            "slam_max_coverage_mean"
                        ],
                        "coverage_min": representative[
                            "slam_max_coverage_min"
                        ],
                        "coverage_p10": representative[
                            "slam_max_coverage_p10"
                        ],
                        "query_count": representative[
                            "slam_max_coverage_query_count"
                        ],
                        "geometry_weight": representative[
                            "slam_max_coverage_geometry_weight"
                        ],
                        "visual_weight": representative[
                            "slam_max_coverage_visual_weight"
                        ],
                        "visual_source": representative[
                            "slam_max_coverage_visual_source"
                        ],
                        "retained_memory_frames": [
                            int(frame_idx)
                            for frame_idx in retained_frames_after_update
                        ],
                    }
                )
            for evicted_frame_idx in evicted_frames:
                score_detail = eviction_score_details.get(evicted_frame_idx, {})
                all_generated_frames.pop(evicted_frame_idx, None)
                write_access_trace(
                    {
                        "event": "memory_eviction",
                        "section_idx": section_idx,
                        "evicted_memory_frame": int(evicted_frame_idx),
                        "section_end_frame": int(section_end_frame),
                        "memory_age_at_eviction": int(section_end_frame - evicted_frame_idx),
                        "stored_memory_size": len(memory_buffer),
                        "memory_policy": memory_policy,
                        "memory_budget": memory_budget,
                        "memory_bank_device": memory_bank_device,
                        "eviction_score": score_detail.get("score"),
                        "eviction_rarity": score_detail.get("rarity"),
                        "eviction_irreplaceability": score_detail.get("irreplaceability"),
                        "eviction_cluster_id": score_detail.get("cluster_id"),
                        "eviction_cluster_size": score_detail.get("cluster_size"),
                        "eviction_dino_cluster_threshold": score_detail.get("cluster_threshold"),
                        "eviction_rgb_nearest_frame": score_detail.get("rgb_nearest_frame"),
                        "eviction_rgb_nearest_distance": score_detail.get("rgb_nearest_distance"),
                        "eviction_redundancy_ratio": score_detail.get("redundancy_ratio"),
                        "eviction_covisible_observers": score_detail.get("covisible_observers"),
                        "eviction_max_covisibility": score_detail.get("max_covisibility"),
                        "eviction_nearest_covisible_frame": score_detail.get("nearest_covisible_frame"),
                        "eviction_marginal_contribution": score_detail.get("marginal_contribution"),
                        "eviction_unique_bonus": score_detail.get("unique_bonus"),
                        "eviction_slamri_beta": score_detail.get("slamri_beta"),
                        "eviction_slamri_forced_keep": score_detail.get("slamri_forced_keep"),
                        "eviction_slamri_slam_raw": score_detail.get("slamri_slam_raw"),
                        "eviction_slamri_slam_norm": score_detail.get("slamri_slam_norm"),
                        "eviction_slamri_ri_raw": score_detail.get("slamri_ri_raw"),
                        "eviction_slamri_ri_norm": score_detail.get("slamri_ri_norm"),
                        "eviction_slamri_ri_rarity": score_detail.get("slamri_ri_rarity"),
                        "eviction_slamri_ri_irreplaceability": score_detail.get("slamri_ri_irreplaceability"),
                        "eviction_slamri_slam_redundancy_ratio": score_detail.get("slamri_slam_redundancy_ratio"),
                        "eviction_slamri_slam_unique_bonus": score_detail.get("slamri_slam_unique_bonus"),
                        "eviction_slam_max_coverage_selected": score_detail.get("slam_max_coverage_selected"),
                        "eviction_slam_max_coverage_forced_keep": score_detail.get("slam_max_coverage_forced_keep"),
                        "eviction_slam_max_coverage_rank": score_detail.get("slam_max_coverage_rank"),
                        "eviction_slam_max_coverage_marginal_gain": score_detail.get("slam_max_coverage_marginal_gain"),
                        "eviction_slam_max_coverage_candidate_gain": score_detail.get("slam_max_coverage_candidate_gain"),
                        "eviction_slam_max_coverage_removal_loss": score_detail.get("slam_max_coverage_removal_loss"),
                        "eviction_slam_max_coverage_value": score_detail.get("slam_max_coverage_value"),
                        "eviction_slam_max_coverage_affinity_mean": score_detail.get("slam_max_coverage_affinity_mean"),
                        "eviction_slam_max_coverage_affinity_max": score_detail.get("slam_max_coverage_affinity_max"),
                        "eviction_coreset_selected": score_detail.get("coreset_selected"),
                        "eviction_coreset_forced_keep": score_detail.get("coreset_forced_keep"),
                        "eviction_coreset_rank": score_detail.get("coreset_rank"),
                        "eviction_coreset_marginal_gain": score_detail.get("coreset_marginal_gain"),
                        "eviction_coreset_candidate_gain": score_detail.get("coreset_candidate_gain"),
                        "eviction_coreset_removal_loss": score_detail.get("coreset_removal_loss"),
                        "eviction_coreset_archive_size": score_detail.get("coreset_archive_size"),
                        "eviction_coreset_facility_value": score_detail.get("coreset_facility_value"),
                        "eviction_coreset_quality": score_detail.get("coreset_quality"),
                        "eviction_coreset_similarity_mean": score_detail.get("coreset_similarity_mean"),
                        "eviction_coreset_similarity_max": score_detail.get("coreset_similarity_max"),
                        "eviction_kcenter_selected": score_detail.get("kcenter_selected"),
                        "eviction_kcenter_forced_keep": score_detail.get("kcenter_forced_keep"),
                        "eviction_kcenter_rank": score_detail.get("kcenter_rank"),
                        "eviction_kcenter_radius": score_detail.get("kcenter_radius"),
                        "eviction_kcenter_mean_radius": score_detail.get("kcenter_mean_radius"),
                        "eviction_kcenter_removal_radius_increase": score_detail.get("kcenter_removal_radius_increase"),
                        "eviction_kcenter_archive_size": score_detail.get("kcenter_archive_size"),
                        "eviction_kcenter_nearest_archive_frame": score_detail.get("kcenter_nearest_archive_frame"),
                        "eviction_kcenter_nearest_archive_distance": score_detail.get("kcenter_nearest_archive_distance"),
                        "eviction_kcenter_selected_for_archive_frame": score_detail.get("kcenter_selected_for_archive_frame"),
                        "eviction_trajectory_selected": score_detail.get("trajectory_selected"),
                        "eviction_trajectory_forced_keep": score_detail.get("trajectory_forced_keep"),
                        "eviction_trajectory_rank": score_detail.get("trajectory_rank"),
                        "eviction_trajectory_marginal_gain": score_detail.get("trajectory_marginal_gain"),
                        "eviction_trajectory_candidate_gain": score_detail.get("trajectory_candidate_gain"),
                        "eviction_trajectory_removal_loss": score_detail.get("trajectory_removal_loss"),
                        "eviction_trajectory_archive_size": score_detail.get("trajectory_archive_size"),
                        "eviction_trajectory_value": score_detail.get("trajectory_value"),
                        "eviction_trajectory_coverage_mean": score_detail.get("trajectory_coverage_mean"),
                        "eviction_trajectory_coverage_min": score_detail.get("trajectory_coverage_min"),
                        "eviction_trajectory_coverage_p10": score_detail.get("trajectory_coverage_p10"),
                        "eviction_trajectory_similarity_mean": score_detail.get("trajectory_similarity_mean"),
                        "eviction_trajectory_similarity_max": score_detail.get("trajectory_similarity_max"),
                        "eviction_density_coverage_selected": score_detail.get("density_coverage_selected"),
                        "eviction_density_coverage_forced_keep": score_detail.get("density_coverage_forced_keep"),
                        "eviction_density_coverage_rank": score_detail.get("density_coverage_rank"),
                        "eviction_density_coverage_marginal_gain": score_detail.get("density_coverage_marginal_gain"),
                        "eviction_density_coverage_candidate_gain": score_detail.get("density_coverage_candidate_gain"),
                        "eviction_density_coverage_removal_loss": score_detail.get("density_coverage_removal_loss"),
                        "eviction_density_coverage_base_mass": score_detail.get("density_coverage_base_mass"),
                        "eviction_density_coverage_assigned_mass": score_detail.get("density_coverage_assigned_mass"),
                        "eviction_density_coverage_total_mass": score_detail.get("density_coverage_total_mass"),
                        "eviction_density_coverage_density": score_detail.get("density_coverage_density"),
                        "eviction_density_coverage_demand_weight": score_detail.get("density_coverage_demand_weight"),
                        "eviction_density_coverage_value": score_detail.get("density_coverage_value"),
                        "eviction_density_coverage_mean": score_detail.get("density_coverage_mean"),
                        "eviction_density_coverage_min": score_detail.get("density_coverage_min"),
                        "eviction_density_coverage_p10": score_detail.get("density_coverage_p10"),
                        "eviction_density_coverage_geometry_mean": score_detail.get("density_coverage_geometry_mean"),
                        "eviction_density_coverage_dino_mean": score_detail.get("density_coverage_dino_mean"),
                        "eviction_density_coverage_rgb_distance_mean": score_detail.get("density_coverage_rgb_distance_mean"),
                        "eviction_density_coverage_kernel_mean": score_detail.get("density_coverage_kernel_mean"),
                        "eviction_density_coverage_alpha": score_detail.get("density_coverage_alpha"),
                        "eviction_density_coverage_dino_weight": score_detail.get("density_coverage_dino_weight"),
                        "eviction_density_coverage_rgb_weight": score_detail.get("density_coverage_rgb_weight"),
                        "eviction_future_view_coverage_selected": score_detail.get("future_view_coverage_selected"),
                        "eviction_future_view_coverage_forced_keep": score_detail.get("future_view_coverage_forced_keep"),
                        "eviction_future_view_coverage_rank": score_detail.get("future_view_coverage_rank"),
                        "eviction_future_view_coverage_marginal_gain": score_detail.get("future_view_coverage_marginal_gain"),
                        "eviction_future_view_coverage_candidate_gain": score_detail.get("future_view_coverage_candidate_gain"),
                        "eviction_future_view_coverage_removal_loss": score_detail.get("future_view_coverage_removal_loss"),
                        "eviction_future_view_coverage_assigned_mass": score_detail.get("future_view_coverage_assigned_mass"),
                        "eviction_future_view_coverage_base_mass": score_detail.get("future_view_coverage_base_mass"),
                        "eviction_future_view_coverage_total_mass": score_detail.get("future_view_coverage_total_mass"),
                        "eviction_future_view_coverage_value": score_detail.get("future_view_coverage_value"),
                        "eviction_future_view_coverage_future_value": score_detail.get("future_view_coverage_future_value"),
                        "eviction_future_view_coverage_future_query_count": score_detail.get("future_view_coverage_future_query_count"),
                        "eviction_future_view_coverage_future_kernel_mean": score_detail.get("future_view_coverage_future_kernel_mean"),
                        "eviction_future_view_coverage_alpha": score_detail.get("future_view_coverage_alpha"),
                        "eviction_future_view_coverage_dino_weight": score_detail.get("future_view_coverage_dino_weight"),
                        "eviction_future_view_coverage_rgb_weight": score_detail.get("future_view_coverage_rgb_weight"),
                        "eviction_future_view_coverage_future_query_weight": score_detail.get("future_view_coverage_future_query_weight"),
                        "eviction_mce_selected": score_detail.get("mce_selected"),
                        "eviction_mce_forced_keep": score_detail.get("mce_forced_keep"),
                        "eviction_mce_removal_rank": score_detail.get("mce_removal_rank"),
                        "eviction_mce_removal_marginal": score_detail.get("mce_removal_marginal"),
                        "eviction_mce_survivor_marginal": score_detail.get("mce_survivor_marginal"),
                        "eviction_mce_coverage_value": score_detail.get("mce_coverage_value"),
                        "eviction_mce_alpha": score_detail.get("mce_alpha"),
                        "eviction_mce_lambda": score_detail.get("mce_lambda"),
                        "eviction_mce_gamma": score_detail.get("mce_gamma"),
                        "eviction_mce_rarity_neighbors": score_detail.get("mce_rarity_neighbors"),
                        "eviction_mce_num_hist_queries": score_detail.get("mce_num_hist_queries"),
                        "eviction_mce_num_ctrl_queries": score_detail.get("mce_num_ctrl_queries"),
                        "eviction_h2o_read_weight": score_detail.get("h2o_read_weight"),
                        "eviction_h2o_selected_count": score_detail.get("h2o_selected_count"),
                        "eviction_h2o_best_overlap": score_detail.get("h2o_best_overlap"),
                        "eviction_h2o_heavy_score": score_detail.get("h2o_heavy_score"),
                        "eviction_h2o_heavy_rank": score_detail.get("h2o_heavy_rank"),
                        "eviction_h2o_recent_keep": score_detail.get("h2o_recent_keep"),
                        "eviction_h2o_recency_rank": score_detail.get("h2o_recency_rank"),
                        "eviction_h2o_recency_budget": score_detail.get("h2o_recency_budget"),
                        "eviction_surprise_priority": score_detail.get("surprise_priority"),
                        "eviction_surprise_replaced_by": score_detail.get("surprise_replaced_by"),
                    }
                )
                if memory_policy not in {"facility_coreset", "kcenter_coreset"}:
                    memory_dino_features.pop(evicted_frame_idx, None)
                    memory_rgb_features.pop(evicted_frame_idx, None)
                    memory_quality_scores.pop(evicted_frame_idx, None)
            profiler.end_phase("memory_policy_update", policy_phase_start)

            # Store only frames that survived the policy update. Each frame owns its
            # storage so evicting one frame releases that frame rather than retaining
            # an entire decoded section through a tensor view.
            bank_store_phase_start = profiler.start_phase()
            retained_frames = set(memory_buffer.candidates())
            if memory_policy == "surprise_forcing":
                # The newest decoded frame is a one-frame rolling anchor, separate
                # from the fixed-capacity external bank, and is released next section.
                retained_frames.add(section_end_frame)
            bank_source_frames = (
                section_frames if memory_bank_device == "cuda" else section_frames_cpu
            )
            for local_frame_idx in range(bank_source_frames.shape[2]):
                global_frame_idx = section_start_frame + local_frame_idx
                if global_frame_idx not in retained_frames:
                    continue
                all_generated_frames[global_frame_idx] = (
                    bank_source_frames[
                        :,
                        :,
                        local_frame_idx : local_frame_idx + 1,
                        :,
                        :,
                    ]
                    .detach()
                    .clone()
                )
            profiler.end_phase("bank_store", bank_store_phase_start)

            if evicted_frames:
                print(f"Evicted frames: {evicted_frames}")
            print(f"Memory stored frames after section {section_idx}: {len(memory_buffer)}")
            print(f"Section {section_idx} completed.")

            bank_frame_bytes = tensor_mapping_nbytes(all_generated_frames)
            bank_feature_bytes = (
                numpy_mapping_nbytes(memory_dino_features)
                + numpy_mapping_nbytes(memory_rgb_features)
                + numpy_mapping_nbytes(surprise_value_features)
                + 8 * len(memory_quality_scores)
                + 8 * len(memory_coverage_masses)
            )
            duration_sec = profile_metadata.get("duration_sec")
            if duration_sec is not None and total_frames > 1:
                generated_seconds = float(duration_sec) * section_end_frame / (total_frames - 1)
            else:
                generated_seconds = section_end_frame / 30.0

            del section_frames
            profiler.end_section(
                section_end_frame=section_end_frame,
                generated_seconds=generated_seconds,
                stored_memory_size=len(memory_buffer),
                candidate_count=section_candidate_count,
                bank_frame_bytes=bank_frame_bytes,
                bank_feature_bytes=bank_feature_bytes,
                host_to_device_bytes=section_host_to_device_bytes,
            )

        write_access_trace(
            {
                "event": "rollout_complete",
                "generated_sections": int(generated_section_count),
                "last_generated_section": int(generated_section_count - 1),
                "context_override_count": len(context_selection_overrides),
                "memory_policy": memory_policy,
                "memory_budget": memory_budget,
                "memory_bank_device": memory_bank_device,
            }
        )

        # The memory bank is no longer needed once the rollout is complete. Output
        # sections were already decoded and moved to CPU during generation.
        all_generated_frames.clear()
        memory_dino_features.clear()
        memory_rgb_features.clear()
        memory_quality_scores.clear()
        memory_coverage_masses.clear()
        surprise_value_features.clear()
        all_section_latents.clear()
        visual_feature_extractor = None

        # Convert one decoded section at a time. Concatenating the full rollout
        # creates another 180-second tensor, and tensor2video then creates a
        # full-size float copy; together those temporaries can exhaust host RAM.
        frames = []
        for section_idx, output_section in enumerate(output_frame_sections):
            frames.extend(self.tensor2video(output_section))
            output_frame_sections[section_idx] = None
            del output_section
        output_frame_sections.clear()
        profiler.finish_rollout()
        write_attention_audit(
            {
                "event": "attention_audit_complete",
                "probe_event_count": int(attention_audit_event_count),
                "probe_sections": sorted(probe_sections),
                "probe_steps": sorted(probe_steps),
                "probe_layers": probe_layers,
            }
        )
        if access_trace_handle is not None:
            access_trace_handle.close()
        if attention_audit_handle is not None:
            attention_audit_handle.close()
        self.load_models_to_device([])
        return frames
