"""
HunyuanWorld-Mirror Inference Pipeline

Usage:
    # Python API — Single GPU
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
    pipeline = WorldMirrorPipeline.from_pretrained('tencent/HY-World-2.0')
    result = pipeline('path/to/images')

    # Python API — Multi-GPU (in a torchrun script)
    pipeline = WorldMirrorPipeline.from_pretrained(
        'tencent/HY-World-2.0', use_fsdp=True, enable_bf16=True)
    result = pipeline('path/to/images')

    # CLI — Single GPU
    python -m hyworld2.worldrecon.pipeline --input_path path/to/images

    # CLI — Multi-GPU
    torchrun --nproc_per_node=2 -m hyworld2.worldrecon.pipeline --input_path path/to/images --use_fsdp --enable_bf16
"""

import argparse
import functools
import gc
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .hyworldmirror.models.models.worldmirror import WorldMirror
from .hyworldmirror.models.layers.block import Block, DistBlock
from .hyworldmirror.models.heads.dense_head import DPTHead
from .hyworldmirror.models.heads.camera_head import CameraHead
from .hyworldmirror.utils.inference_utils import (
    prepare_images_to_tensor,
    prepare_input,
    compute_adaptive_target_size,
    compute_preprocessing_transform,
    load_prior_camera,
    load_prior_depth,
    compute_sky_mask,
    compute_filter_mask,
    save_results,
    print_and_save_timings,
)
from .hyworldmirror.utils.render_utils import render_interpolated_video


# ============================================================
# Model loading helpers (checkpoint, config, selective load)
# ============================================================

def _get_model_config_from_yaml(cfg) -> dict:
    if hasattr(cfg, "wrapper") and hasattr(cfg.wrapper, "model"):
        model_cfg = cfg.wrapper.model
    elif hasattr(cfg, "model"):
        model_cfg = cfg.model
    else:
        raise ValueError("No model config found (expect wrapper.model or model).")
    out = OmegaConf.to_container(model_cfg, resolve=True)
    out.pop("_target_", None)
    return out


def _load_checkpoint_state_dict(ckpt_path: str) -> dict:
    if ckpt_path.endswith(".safetensors"):
        return load_safetensors(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if "state_dict" in ckpt:
        state = {k.replace("model.", ""): v for k, v in state.items()}
    return state


def _load_state_dict_selective(model, ckpt_state, source_name="checkpoint"):
    current = model.state_dict()
    for key in current:
        if key in ckpt_state and current[key].shape == ckpt_state[key].shape:
            current[key] = ckpt_state[key]
    model.load_state_dict(current, strict=True)
    matched = sum(1 for k in current if k in ckpt_state and current[k].shape == ckpt_state[k].shape)
    print(f"  Loaded {matched}/{len(current)} keys from {source_name}")


def _has_model_files(path: str) -> bool:
    """Check whether a directory contains the expected model artifacts."""
    has_weights = os.path.isfile(os.path.join(path, "model.safetensors"))
    has_config = (os.path.isfile(os.path.join(path, "config.yaml"))
                  or os.path.isfile(os.path.join(path, "config.json")))
    return has_weights and has_config


def _resolve_model_dir(model_path: str, subfolder: str) -> str:
    """Resolve a local directory containing config + model.safetensors.

    Resolution order:
      1. {model_path}/{subfolder}  — local repo root with subfolder
      2. {model_path}              — direct local path (backward compat)
      3. HuggingFace download: snapshot_download(repo_id, allow_patterns=[subfolder/*])
    """
    candidate = os.path.join(model_path, subfolder)
    if os.path.isdir(candidate) and _has_model_files(candidate):
        print(f"[Init] Found local model at {candidate}")
        return candidate

    if os.path.isdir(model_path) and _has_model_files(model_path):
        print(f"[Init] Found local model at {model_path}")
        return model_path

    print(f"[Init] Downloading from HuggingFace: {model_path} (subfolder={subfolder})")
    from huggingface_hub import snapshot_download
    repo_root = snapshot_download(
        repo_id=model_path,
        allow_patterns=[f"{subfolder}/*"],
    )
    resolved = os.path.join(repo_root, subfolder)
    if not _has_model_files(resolved):
        raise FileNotFoundError(
            f"Downloaded repo '{model_path}' but subfolder '{subfolder}' "
            f"does not contain model.safetensors + config. "
            f"Check that the repo and subfolder name are correct."
        )
    return resolved


def _load_model_config(model_dir: str) -> dict:
    """Load model constructor kwargs from config.yaml or config.json in model_dir."""
    import json as _json
    yaml_path = os.path.join(model_dir, "config.yaml")
    json_path = os.path.join(model_dir, "config.json")

    if os.path.isfile(yaml_path):
        cfg = OmegaConf.load(yaml_path)
        return _get_model_config_from_yaml(cfg)
    elif os.path.isfile(json_path):
        with open(json_path) as f:
            return _json.load(f)
    else:
        raise FileNotFoundError(f"No config.yaml or config.json in {model_dir}")


# ============================================================
# FSDP / bf16 helpers
# ============================================================

def _collect_fp32_critical_modules(model):
    from .hyworldmirror.models.layers.mlp import MlpFP32
    critical = set()
    for _, module in model.named_modules():
        if isinstance(module, MlpFP32) and hasattr(module, 'fc2'):
            if any(p.dtype == torch.float32 for p in module.fc2.parameters()):
                critical.add(module.fc2)
        if hasattr(module, 'scratch') and hasattr(module.scratch, 'output_conv2'):
            oc2 = module.scratch.output_conv2
            if any(p.dtype == torch.float32 for p in oc2.parameters()):
                critical.add(oc2)
    return critical


def _cast_noncritical_fp32_to_bf16(model, critical_modules):
    critical_ids = {id(p) for mod in critical_modules for p in mod.parameters()}
    cast = []
    for name, param in model.named_parameters():
        if param.dtype == torch.float32 and id(param) not in critical_ids:
            param.data = param.data.to(torch.bfloat16)
            cast.append(name)
    for _, buf in model.named_buffers():
        if buf.dtype == torch.float32:
            buf.data = buf.data.to(torch.bfloat16)

    def _hook(module, args):
        if not args:
            return args
        dtype = next((p.dtype for p in module.parameters(recurse=False)), None)
        if dtype is None:
            return args
        return tuple(a.to(dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != dtype else a
                     for a in args)

    for name, module in model.named_modules():
        if not any(True for _ in module.children()):
            own = list(module.named_parameters(recurse=False))
            if own and all(p.dtype == torch.bfloat16 for _, p in own):
                pfx = name + "." if name else ""
                if any(c.startswith(pfx) for c in cast):
                    module.register_forward_pre_hook(_hook)


def _wrap_model_fsdp(model, sp_group, device, use_cpu_offload=False, enable_bf16=False):
    wrap_cls = {DistBlock, Block, DPTHead, CameraHead}
    if enable_bf16:
        fp32_critical = _collect_fp32_critical_modules(model)
        def policy(module, recurse, nonwrapped_numel, **kw):
            if recurse:
                return True
            return isinstance(module, tuple(wrap_cls)) or module in fp32_critical
        auto_wrap_policy = policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=wrap_cls)

    fsdp_model = FSDP(
        model, process_group=sp_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy, mixed_precision=None,
        cpu_offload=CPUOffload(offload_params=True) if use_cpu_offload else None,
        device_id=device, use_orig_params=True, sync_module_states=True,
        forward_prefetch=False,
    )

    rank = dist.get_rank()
    if rank == 0:
        total = sum(p.numel() for p in fsdp_model.parameters())
        local = sum(getattr(p, '_local_tensor', p).numel() for p in fsdp_model.parameters())
        print(f"[FSDP] total={total/1e6:.1f}M, local≈{local/1e6:.1f}M")
    return fsdp_model


# ============================================================
# WorldMirrorPipeline
# ============================================================

class WorldMirrorPipeline:
    """HunyuanWorld-Mirror inference pipeline.

    Supports single-GPU and multi-GPU (Sequence Parallel) inference with
    a unified API. Multi-GPU mode is auto-detected from torch.distributed.
    """

    def __init__(self, model, device, sp_size=1, sp_group=None, rank=0):
        self.model = model
        self.device = device
        self.sp_size = sp_size
        self.sp_group = sp_group
        self.rank = rank

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "tencent/HY-World-2.0",
        *,
        subfolder: str = "HY-WorldMirror-2.0",
        config_path: str = None,
        ckpt_path: str = None,
        use_fsdp: bool = False,
        enable_bf16: bool = False,
        fsdp_cpu_offload: bool = False,
        disable_heads: list = None,
    ) -> "WorldMirrorPipeline":
        """Load model and create pipeline instance.

        Automatically detects distributed mode (torchrun sets WORLD_SIZE).

        Args:
            pretrained_model_name_or_path: HuggingFace repo ID or local path.
                The model files are expected under ``{path}/{subfolder}/``.
            subfolder: Subfolder inside the repo that contains the WorldMirror
                checkpoint (model.safetensors + config).
            config_path: Training config YAML (used with ckpt_path).
            ckpt_path: Checkpoint file (.ckpt / .safetensors).
            use_fsdp: Shard parameters across GPUs via FSDP.
            enable_bf16: Use bf16 precision (except critical layers).
            fsdp_cpu_offload: Offload FSDP params to CPU.
            disable_heads: List of heads to disable, e.g. ["camera", "depth"].
        """
        is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

        if is_distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            sp_size = world_size
            sp_group = dist.new_group(ranks=list(range(sp_size)))
            if rank == 0:
                print(f"[Pipeline] Multi-GPU: world_size={world_size}, sp_size={sp_size}")
        else:
            rank, sp_size = 0, 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sp_group = None
            if use_fsdp:
                print("[Pipeline] Warning: use_fsdp is ignored in single-GPU mode (FSDP requires torchrun with multiple GPUs)")
                use_fsdp = False
            print("[Pipeline] Single-GPU mode")

        # Load model
        t0 = time.perf_counter()
        if config_path and ckpt_path:
            print(f"[Init] config={config_path}, ckpt={ckpt_path}, sp_size={sp_size}")
            cfg = OmegaConf.load(config_path)
            model_cfg = _get_model_config_from_yaml(cfg)
            if sp_size > 1:
                model_cfg["sp_size"] = sp_size
            if enable_bf16:
                model_cfg["enable_bf16"] = True
            model = WorldMirror(**model_cfg).to(device)
            state = _load_checkpoint_state_dict(ckpt_path)
            _load_state_dict_selective(model, state, source_name=ckpt_path)
            del state; gc.collect(); torch.cuda.empty_cache()
        else:
            model_dir = _resolve_model_dir(pretrained_model_name_or_path, subfolder)
            model_cfg = _load_model_config(model_dir)
            if sp_size > 1:
                model_cfg["sp_size"] = sp_size
            if enable_bf16:
                model_cfg["enable_bf16"] = True
            model = WorldMirror(**model_cfg).to(device)
            state = load_safetensors(os.path.join(model_dir, "model.safetensors"))
            _load_state_dict_selective(model, state, source_name=model_dir)
            del state; gc.collect(); torch.cuda.empty_cache()

        # bf16 casting — two strategies depending on FSDP:
        #
        # Multi-GPU + FSDP: cast everything to bf16 uniformly (including fc2).
        #   FSDP requires uniform dtype per flat-param unit.
        #
        # Single GPU (no FSDP): cast to bf16, then restore critical fp32
        #   modules (MlpFP32.fc2, output_conv2) so their .float() calls work.
        #   Register input-cast hooks on bf16 leaf modules for dtype boundaries.
        if enable_bf16:
            if use_fsdp and is_distributed:
                model.to(torch.bfloat16)
                crit = _collect_fp32_critical_modules(model)
                _cast_noncritical_fp32_to_bf16(model, crit)
            else:
                crit = _collect_fp32_critical_modules(model)
                model.to(torch.bfloat16)
                for mod in crit:
                    mod.to(torch.float32)

                def _input_cast_hook(module, args):
                    if not args:
                        return args
                    dtype = next((p.dtype for p in module.parameters(recurse=False)), None)
                    if dtype is None:
                        return args
                    return tuple(
                        a.to(dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != dtype else a
                        for a in args
                    )

                for _, module in model.named_modules():
                    if not any(True for _ in module.children()):
                        own = list(module.parameters(recurse=False))
                        if own and all(p.dtype == torch.bfloat16 for p in own):
                            module.register_forward_pre_hook(_input_cast_hook)

        model.eval()

        # Disable unused heads
        if disable_heads:
            _disable_heads(model, disable_heads)

        # FSDP wrapping
        if use_fsdp and is_distributed:
            model = _wrap_model_fsdp(model, sp_group, device,
                                     use_cpu_offload=fsdp_cpu_offload,
                                     enable_bf16=enable_bf16)
            if enable_bf16:
                inner = model.module if hasattr(model, 'module') else model
                inner.to = lambda *a, **kw: inner

        if rank == 0:
            print(f"[Init] Model ready in {time.perf_counter() - t0:.1f}s")
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(device) / (1024**3)
                print(f"[Memory] allocated={alloc:.2f}GB")

        return cls(model, device, sp_size, sp_group, rank)

    @torch.no_grad()
    def __call__(
        self,
        input_path: str,
        output_path: str = "inference_output",
        *,
        # Inference
        target_size: int = 952,
        fps: int = 1,
        video_strategy: str = "new",
        video_min_frames: int = 1,
        video_max_frames: int = 32,
        # Save
        save_depth: bool = True,
        save_normal: bool = True,
        save_gs: bool = True,
        save_camera: bool = True,
        save_points: bool = True,
        save_colmap: bool = False,
        save_conf: bool = False,
        # Mask
        apply_sky_mask: bool = True,
        apply_edge_mask: bool = True,
        apply_confidence_mask: bool = False,
        # NegenWM-JEPA-v2 : NER mask (use_ner=False — zéro breaking change)
        use_ner: bool = False,
        ner_tau: float = 0.35,
        ner_lambda_sky: float = 0.30,
        ner_lambda_edge: float = 0.30,
        save_sky_mask: bool = False,
        sky_mask_source: str = "auto",
        model_sky_threshold: float = 0.45,
        confidence_percentile: float = 10.0,
        edge_normal_threshold: float = 1.0,
        edge_depth_threshold: float = 0.03,
        # Compression
        compress_pts: bool = True,
        compress_pts_max_points: int = 2_000_000,
        compress_pts_voxel_size: float = 0.002,
        max_resolution: int = 1920,
        compress_gs_max_points: int = 5_000_000,
        # Prior
        prior_cam_path: str = None,
        prior_depth_path: str = None,
        # Rendered video
        save_rendered: bool = False,
        render_interp_per_pair: int = 15,
        render_depth: bool = False,
        # Misc
        log_time: bool = True,
        strict_output_path: str = None,
    ) -> str:
        """Run inference on images/video and save results.

        Args:
            input_path: Directory of images or a video file.
            output_path: Root output directory.
            **kwargs: Override default inference parameters.

        Returns:
            Path to the output directory (str), or None on skip.
        """
        model = self.model
        device = self.device
        sp_size, sp_group, rank = self.sp_size, self.sp_group, self.rank
        is_distributed = sp_size > 1

        case_t0 = time.perf_counter()
        timings = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Prepare input
        t0 = time.perf_counter()
        img_paths, subdir_name = prepare_input(
            input_path, target_size=target_size, fps=fps,
            video_strategy=video_strategy,
            min_frames=video_min_frames, max_frames=video_max_frames,
        )
        if log_time:
            timings["data_loading"] = time.perf_counter() - t0

        if strict_output_path is not None:
            outdir = Path(strict_output_path)
        else:
            outdir = Path(output_path) / subdir_name / timestamp

        # 2. Adaptive resolution
        effective = compute_adaptive_target_size(img_paths, target_size)
        if rank == 0 and effective != target_size:
            print(f"[Inference] Adaptive resolution: {effective} (max={target_size})")

        # 3. Inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t0_all = time.perf_counter()
        try:
            predictions, imgs, infer_time = self._run_inference(
                img_paths, effective, prior_cam_path, prior_depth_path)
        except ValueError as e:
            if rank == 0:
                print(f"[Pipeline] Skipping '{input_path}': {e}")
            return None

        if log_time:
            timings["inference"] = infer_time
            timings["inference_preprocess"] = time.perf_counter() - t0_all - infer_time

        # GPU memory stats (multi-GPU)
        if log_time and torch.cuda.is_available() and is_distributed:
            peak = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_t = torch.tensor([peak], dtype=torch.float64, device=device)
            gathered = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(sp_size)]
            dist.all_gather(gathered, peak_t, group=sp_group)
            timings["gpu_mem_peak_per_rank_gb"] = [t.item() for t in gathered]
            timings["gpu_mem_peak_avg_gb"] = sum(timings["gpu_mem_peak_per_rank_gb"]) / sp_size

        # 4. Post-processing and saving (rank 0 only)
        if rank == 0:
            B, S, C, H, W = imgs.shape
            t0 = time.perf_counter()

            sky_mask = (compute_sky_mask(
                img_paths, H, W, S, predictions=predictions,
                source=sky_mask_source, model_threshold=model_sky_threshold,
                processed_aspect_ratio=W / H,
            ) if apply_sky_mask else None)

            filter_mask, gs_filter_mask = None, None
            if apply_confidence_mask or apply_edge_mask or apply_sky_mask:
                filter_mask, gs_filter_mask = compute_filter_mask(
                    predictions, imgs, img_paths, H, W, S,
                    apply_confidence_mask=apply_confidence_mask,
                    apply_edge_mask=apply_edge_mask,
                    apply_sky_mask=apply_sky_mask,
                    confidence_percentile=confidence_percentile,
                    edge_normal_threshold=edge_normal_threshold,
                    edge_depth_threshold=edge_depth_threshold,
                    sky_mask=sky_mask, use_gs_depth=save_gs,
                    use_ner=use_ner, ner_tau=ner_tau,
                    ner_lambda_sky=ner_lambda_sky, ner_lambda_edge=ner_lambda_edge,
                )

            if log_time:
                timings["compute_mask"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            save_timings = save_results(
                predictions, imgs, img_paths, outdir,
                save_depth=save_depth, save_normal=save_normal,
                save_gs=save_gs, save_camera=save_camera,
                save_points=save_points, save_colmap=save_colmap,
                save_sky_mask=save_sky_mask, save_conf=save_conf,
                log_time=log_time, max_resolution=max_resolution,
                filter_mask=filter_mask, gs_filter_mask=gs_filter_mask,
                sky_mask=sky_mask,
                compress_pts=compress_pts,
                compress_pts_max_points=compress_pts_max_points,
                compress_pts_voxel_size=compress_pts_voxel_size,
                compress_gs_max_points=compress_gs_max_points,
            )
            if log_time:
                timings.update(save_timings or {})
                timings["save_total_wall"] = time.perf_counter() - t0

            # Render interpolated video from Gaussian splats
            if save_rendered and "splats" in predictions:
                inner_model = model.module if hasattr(model, 'module') else model
                if hasattr(inner_model, 'gs_renderer'):
                    t0_render = time.perf_counter()
                    try:
                        splats_f32 = {k: v.float() if isinstance(v, torch.Tensor) else v
                                      for k, v in predictions["splats"].items()}
                        render_interpolated_video(
                            inner_model.gs_renderer,
                            splats_f32,
                            predictions["camera_poses"].float(),
                            predictions["camera_intrs"].float(),
                            (H, W),
                            outdir / "rendered",
                            interp_per_pair=render_interp_per_pair,
                            loop_reverse=(S <= 2),
                            render_depth=render_depth,
                        )
                        if log_time:
                            timings["render_video"] = time.perf_counter() - t0_render
                    except Exception as e:
                        print(f"[Pipeline] Warning: video rendering failed: {e}")

            if not is_distributed:
                del predictions
                torch.cuda.empty_cache()

            timings["case_total"] = time.perf_counter() - case_t0
            if log_time:
                print_and_save_timings(timings, outdir)

            print(f"\n{'='*60}\n[Pipeline] Results saved to: {outdir}\n{'='*60}\n")

        if is_distributed:
            del predictions, imgs
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()

        return str(outdir)

    def _run_inference(self, img_paths, target_size, prior_cam_path, prior_depth_path):
        """Run model forward pass."""
        device = self.device
        imgs = prepare_images_to_tensor(
            img_paths, target_size=target_size, resize_strategy="crop"
        ).to(device)
        views = {"img": imgs}
        B, S, C, H, W = imgs.shape

        if self.sp_size > 1 and S < self.sp_size:
            raise ValueError(
                f"Number of input images ({S}) must be >= number of GPUs ({self.sp_size}) "
                f"in multi-GPU mode. Please provide at least {self.sp_size} images, "
                f"or use fewer GPUs."
            )

        if self.rank == 0:
            print(f"[Inference] {S} images, shape={imgs.shape}, sp_size={self.sp_size}")

        pp_xform = compute_preprocessing_transform(img_paths, target_size)
        cond_flags = [0, 0, 0]

        if prior_cam_path and os.path.isfile(prior_cam_path):
            extr, intr = load_prior_camera(prior_cam_path, img_paths, preprocess_transform=pp_xform)
            if extr is not None:
                first = extr[0, 0]
                extr = torch.linalg.inv(first.float()).to(first.dtype).unsqueeze(0).unsqueeze(0) @ extr
                views["camera_poses"] = extr.to(device)
                cond_flags[0] = 1
            if intr is not None:
                views["camera_intrs"] = intr.to(device)
                cond_flags[2] = 1

        if prior_depth_path and os.path.isdir(prior_depth_path):
            depth = load_prior_depth(prior_depth_path, img_paths, H, W, preprocess_transform=pp_xform)
            if depth is not None:
                views["depthmap"] = depth.to(device)
                cond_flags[1] = 1

        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        inner = self.model.module if hasattr(self.model, 'module') else self.model
        model_bf16 = getattr(inner, 'enable_bf16', False)

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(not model_bf16 and use_amp), dtype=torch.bfloat16):
            fwd_kw = dict(views=views, cond_flags=cond_flags, is_inference=True)
            if self.sp_size > 1:
                fwd_kw["sp_size"] = self.sp_size
                fwd_kw["sp_group"] = self.sp_group
            predictions = self.model(**fwd_kw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0

        if self.rank == 0:
            print(f"[Inference] Done in {infer_time:.2f}s")
        return predictions, imgs, infer_time


# ============================================================
# Head disabling helper
# ============================================================

def _disable_heads(model, head_names):
    """Disable and free specified heads. head_names: list of 'camera','depth','normal','points','gs'."""
    mapping = {
        "camera": ("enable_cam",   ["cam_head"]),
        "depth":  ("enable_depth", ["depth_head"]),
        "normal": ("enable_norm",  ["norm_head"]),
        "points": ("enable_pts",   ["pts_head"]),
        "gs":     ("enable_gs",    ["gs_head", "gs_renderer"]),
    }
    freed = 0
    for name in head_names:
        if name not in mapping:
            continue
        attr, modules = mapping[name]
        setattr(model, attr, False)
        for mod_name in modules:
            if hasattr(model, mod_name):
                mod = getattr(model, mod_name)
                freed += sum(p.numel() for p in mod.parameters())
                mod.cpu()
                delattr(model, mod_name)
                del mod
    if freed:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Init] Disabled heads: {head_names}, freed ~{freed/1e6:.1f}M params")


# ============================================================
# CLI entry point
# ============================================================

def _broadcast_string(s, rank, src=0):
    if rank == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device="cuda")
    else:
        length = torch.tensor([0], dtype=torch.long, device="cuda")
    dist.broadcast(length, src=src)
    n = length.item()
    tensor = torch.tensor(list(data), dtype=torch.uint8, device="cuda") if rank == src else torch.empty(n, dtype=torch.uint8, device="cuda")
    dist.broadcast(tensor, src=src)
    return tensor.cpu().numpy().tobytes().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="HunyuanWorld-Mirror Pipeline")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="inference_output")
    parser.add_argument("--strict_output_path", type=str, default=None,
                        help="If set, save results directly to this path (no subdir/timestamp)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="tencent/HY-World-2.0",
                        help="HuggingFace repo ID or local path")
    parser.add_argument("--subfolder", type=str, default="HY-WorldMirror-2.0",
                        help="Subfolder inside the repo containing WorldMirror weights")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--use_fsdp", action="store_true", default=False)
    parser.add_argument("--enable_bf16", action="store_true", default=False)
    parser.add_argument("--fsdp_cpu_offload", action="store_true", default=False)
    parser.add_argument("--target_size", type=int, default=952)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--video_strategy", type=str, default="new", choices=["old", "new"])
    parser.add_argument("--video_min_frames", type=int, default=1)
    parser.add_argument("--video_max_frames", type=int, default=32)
    parser.add_argument("--no_save_depth", action="store_true")
    parser.add_argument("--no_save_normal", action="store_true")
    parser.add_argument("--no_save_gs", action="store_true")
    parser.add_argument("--no_save_camera", action="store_true")
    parser.add_argument("--no_save_points", action="store_true")
    parser.add_argument("--save_colmap", action="store_true", default=False)
    parser.add_argument("--save_conf", action="store_true", default=False)
    parser.add_argument("--save_sky_mask", action="store_true", default=False)
    parser.add_argument("--apply_sky_mask", action="store_true", default=True)
    parser.add_argument("--no_sky_mask", dest="apply_sky_mask", action="store_false")
    parser.add_argument("--apply_edge_mask", action="store_true", default=True)
    parser.add_argument("--no_edge_mask", dest="apply_edge_mask", action="store_false")
    parser.add_argument("--apply_confidence_mask", action="store_true", default=False)
    # NegenWM-JEPA-v2 : NER flags (use_ner=False par défaut — zéro breaking change)
    parser.add_argument("--use_ner", action="store_true", default=False,
                        help="Use NERHead (NegenWM-JEPA-v2) for differentiable point filtering")
    parser.add_argument("--ner_tau", type=float, default=0.35, help="NER binarization threshold")
    parser.add_argument("--ner_lambda_sky", type=float, default=0.30, help="NER sky prior weight")
    parser.add_argument("--ner_lambda_edge", type=float, default=0.30, help="NER edge prior weight")
    parser.add_argument("--sky_mask_source", type=str, default="auto", choices=["auto", "model", "onnx"])
    parser.add_argument("--model_sky_threshold", type=float, default=0.45)
    parser.add_argument("--confidence_percentile", type=float, default=10.0)
    parser.add_argument("--edge_normal_threshold", type=float, default=1.0)
    parser.add_argument("--edge_depth_threshold", type=float, default=0.03)
    parser.add_argument("--compress_pts", action="store_true", default=True)
    parser.add_argument("--no_compress_pts", dest="compress_pts", action="store_false")
    parser.add_argument("--compress_pts_max_points", type=int, default=2_000_000)
    parser.add_argument("--compress_pts_voxel_size", type=float, default=0.002)
    parser.add_argument("--max_resolution", type=int, default=1920)
    parser.add_argument("--compress_gs_max_points", type=int, default=5_000_000)
    parser.add_argument("--prior_cam_path", type=str, default=None)
    parser.add_argument("--prior_depth_path", type=str, default=None)
    parser.add_argument("--disable_heads", type=str, nargs="*", default=None,
                        help="Heads to disable: camera depth normal points gs")
    parser.add_argument("--save_rendered", action="store_true", default=False,
                        help="Render interpolated video from Gaussian splats")
    parser.add_argument("--render_interp_per_pair", type=int, default=15,
                        help="Interpolated frames per camera pair for video rendering")
    parser.add_argument("--render_depth", action="store_true", default=False,
                        help="Also render depth video")
    parser.add_argument("--log_time", action="store_true", default=True)
    parser.add_argument("--no_log_time", dest="log_time", action="store_false")
    parser.add_argument("--no_interactive", action="store_true")
    args = parser.parse_args()

    pipeline = WorldMirrorPipeline.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        subfolder=args.subfolder,
        config_path=args.config_path, ckpt_path=args.ckpt_path,
        use_fsdp=args.use_fsdp, enable_bf16=args.enable_bf16,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        disable_heads=args.disable_heads,
    )

    call_kwargs = dict(
        output_path=args.output_path,
        target_size=args.target_size, fps=args.fps,
        video_strategy=args.video_strategy,
        video_min_frames=args.video_min_frames,
        video_max_frames=args.video_max_frames,
        save_depth=not args.no_save_depth,
        save_normal=not args.no_save_normal,
        save_gs=not args.no_save_gs,
        save_camera=not args.no_save_camera,
        save_points=not args.no_save_points,
        save_colmap=args.save_colmap,
        save_conf=args.save_conf,
        save_sky_mask=args.save_sky_mask,
        apply_sky_mask=args.apply_sky_mask,
        apply_edge_mask=args.apply_edge_mask,
        apply_confidence_mask=args.apply_confidence_mask,
        use_ner=args.use_ner,
        ner_tau=args.ner_tau,
        ner_lambda_sky=args.ner_lambda_sky,
        ner_lambda_edge=args.ner_lambda_edge,
        sky_mask_source=args.sky_mask_source,
        model_sky_threshold=args.model_sky_threshold,
        confidence_percentile=args.confidence_percentile,
        edge_normal_threshold=args.edge_normal_threshold,
        edge_depth_threshold=args.edge_depth_threshold,
        compress_pts=args.compress_pts,
        compress_pts_max_points=args.compress_pts_max_points,
        compress_pts_voxel_size=args.compress_pts_voxel_size,
        max_resolution=args.max_resolution,
        compress_gs_max_points=args.compress_gs_max_points,
        prior_cam_path=args.prior_cam_path,
        prior_depth_path=args.prior_depth_path,
        save_rendered=args.save_rendered,
        render_interp_per_pair=args.render_interp_per_pair,
        render_depth=args.render_depth,
        log_time=args.log_time,
        strict_output_path=args.strict_output_path,
    )

    try:
        pipeline(args.input_path, **call_kwargs)

        if args.no_interactive:
            return

        rank = pipeline.rank
        is_distributed = pipeline.sp_size > 1

        if rank == 0:
            print("\n[Interactive] Enter new input paths. Type 'quit' to stop.\n")

        _INF_TIMEOUT = timedelta(days=365)
        _DEF_TIMEOUT = timedelta(minutes=10)

        while True:
            if is_distributed:
                dist.distributed_c10d._get_default_group()._get_backend(
                    torch.device("cuda")).options._timeout = _INF_TIMEOUT

            new_input = ""
            if rank == 0:
                try:
                    new_input = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):
                    new_input = "quit"

            if is_distributed:
                new_input = _broadcast_string(new_input, rank, src=0)
                dist.distributed_c10d._get_default_group()._get_backend(
                    torch.device("cuda")).options._timeout = _DEF_TIMEOUT

            if not new_input or new_input.lower() in ("quit", "exit", "q"):
                break

            if rank == 0 and not (Path(new_input).is_dir() or Path(new_input).is_file()):
                print(f"  Invalid path: {new_input}")
                continue

            pipeline.model.to(pipeline.device)
            pipeline.model.eval()
            pipeline(new_input, **call_kwargs)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
