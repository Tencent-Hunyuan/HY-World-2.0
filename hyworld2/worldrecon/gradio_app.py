"""
Gradio demo for WorldMirror 2.0 — HunyuanWorld World Reconstruction.

Usage:
    python -m hyworld2.worldrecon.gradio_app
    python -m hyworld2.worldrecon.gradio_app --examples_dir /path/to/examples --port 8081
"""

import argparse
import gc
import io
import os
import shutil
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

try:
    import spaces
except ImportError:
    class _SpacesStub:
        @staticmethod
        def GPU(duration=120):
            def _dec(fn):
                return fn
            return _dec
    spaces = _SpacesStub()

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from .pipeline import WorldMirrorPipeline
from .hyworldmirror.utils.inference_utils import (
    prepare_images_to_tensor,
    compute_adaptive_target_size,
    compute_sky_mask,
    compute_filter_mask,
)
from .hyworldmirror.utils.streaming_save import save_gaussian_splats_artifact
from .hyworldmirror.utils.visual_util import convert_predictions_to_glb_scene
from .hyworldmirror.utils.save_utils import save_camera_params
from .hyworldmirror.models.utils.geometry import depth_to_world_coords_points

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pipeline = None
_pipeline_args = {}
current_terminal_output = ""


def _init_pipeline_args(
    pretrained_model_name_or_path="tencent/HY-World-2.0",
    config_path=None,
    ckpt_path=None,
    use_fsdp=False,
    enable_bf16=False,
    fsdp_cpu_offload=False,
):
    global _pipeline_args
    _pipeline_args = dict(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config_path=config_path,
        ckpt_path=ckpt_path,
        use_fsdp=use_fsdp,
        enable_bf16=enable_bf16,
        fsdp_cpu_offload=fsdp_cpu_offload,
    )


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = WorldMirrorPipeline.from_pretrained(**_pipeline_args)
    elif _pipeline.sp_size <= 1:
        _pipeline.model.to(_pipeline.device)
        _pipeline.model.eval()
    return _pipeline


# ---------------------------------------------------------------------------
# Distributed helpers for multi-GPU Gradio
# ---------------------------------------------------------------------------
def _broadcast_string(s, rank, src=0):
    """Broadcast a string from src to all ranks."""
    import torch.distributed as dist
    if rank == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device="cuda")
    else:
        length = torch.tensor([0], dtype=torch.long, device="cuda")
    dist.broadcast(length, src=src)
    n = length.item()
    if rank == src:
        tensor = torch.tensor(list(data), dtype=torch.uint8, device="cuda")
    else:
        tensor = torch.empty(n, dtype=torch.uint8, device="cuda")
    dist.broadcast(tensor, src=src)
    return tensor.cpu().numpy().tobytes().decode("utf-8")


def _notify_workers(img_paths, target_size):
    """Rank 0 broadcasts inference task to worker ranks."""
    pipe = _get_pipeline()
    if pipe.sp_size <= 1:
        return
    cmd = "__INFER__\t" + str(target_size) + "\t" + "\t".join(img_paths)
    _broadcast_string(cmd, pipe.rank, src=0)


def _worker_loop():
    """Non-rank-0 process loop: wait for inference commands from rank 0."""
    import torch.distributed as dist
    from datetime import timedelta

    pipe = _get_pipeline()
    rank = pipe.rank
    print(f"[Worker rank {rank}] Entering worker loop, waiting for tasks...")

    _INF_TIMEOUT = timedelta(days=365)
    _DEF_TIMEOUT = timedelta(minutes=10)

    while True:
        dist.distributed_c10d._get_default_group()._get_backend(
            torch.device("cuda")).options._timeout = _INF_TIMEOUT

        cmd = _broadcast_string("", rank, src=0)

        dist.distributed_c10d._get_default_group()._get_backend(
            torch.device("cuda")).options._timeout = _DEF_TIMEOUT

        if cmd == "__QUIT__":
            print(f"[Worker rank {rank}] Received quit signal.")
            break

        if cmd.startswith("__INFER__\t"):
            parts = cmd.split("\t")
            target_size = int(parts[1])
            img_paths = parts[2:]
            print(f"[Worker rank {rank}] Running inference ({len(img_paths)} images, size={target_size})")
            with torch.no_grad():
                pipe._run_inference(img_paths, target_size, None, None)

    if dist.is_initialized():
        dist.destroy_process_group()


class TeeOutput:
    """Capture stdout while still printing to console."""

    def __init__(self, max_chars=10000):
        self.terminal = sys.stdout
        self.log = io.StringIO()
        self.max_chars = max_chars

    def write(self, message):
        global current_terminal_output
        self.terminal.write(message)
        self.log.write(message)
        content = self.log.getvalue()
        if len(content) > self.max_chars:
            content = "...(earlier output truncated)...\n" + content[-self.max_chars :]
            self.log = io.StringIO()
            self.log.write(content)
        current_terminal_output = self.log.getvalue()

    def flush(self):
        self.terminal.flush()

    def getvalue(self):
        return self.log.getvalue()

    def clear(self):
        global current_terminal_output
        self.log = io.StringIO()
        current_terminal_output = ""


# ---------------------------------------------------------------------------
# Model inference (uses WorldMirrorPipeline internally)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def run_model(target_dir):
    """Run WorldMirror inference on images inside *target_dir/images*.

    Returns (outputs_dict, processed_visualization_data).
    """
    pipe = _get_pipeline()

    image_folder = os.path.join(target_dir, "images")
    img_paths = sorted(
        glob(os.path.join(image_folder, "*.png"))
        + glob(os.path.join(image_folder, "*.jpg"))
        + glob(os.path.join(image_folder, "*.jpeg"))
        + glob(os.path.join(image_folder, "*.webp"))
    )
    if not img_paths:
        raise ValueError("No images found in the upload directory.")
    if pipe.sp_size > 1 and len(img_paths) < pipe.sp_size:
        raise ValueError(
            f"Multi-GPU mode requires at least {pipe.sp_size} input images "
            f"(got {len(img_paths)}). Please upload more images or use fewer GPUs."
        )

    print(f"[Gradio] {len(img_paths)} images from {image_folder}")

    effective = compute_adaptive_target_size(img_paths, max_target_size=952)

    _notify_workers(img_paths, effective)

    predictions, imgs, infer_time = pipe._run_inference(
        img_paths, effective, prior_cam_path=None, prior_depth_path=None
    )
    B, S, C, H, W = imgs.shape
    print(f"[Gradio] Inference done in {infer_time:.2f}s  ({S} views, {H}x{W})")

    sky_mask = compute_sky_mask(
        img_paths, H, W, S,
        predictions=predictions,
        source="auto",
        model_threshold=0.45,
        processed_aspect_ratio=W / H,
    )

    filter_mask, gs_filter_mask = compute_filter_mask(
        predictions, imgs, img_paths, H, W, S,
        apply_confidence_mask=False,
        apply_edge_mask=True,
        apply_sky_mask=True,
        confidence_percentile=10.0,
        edge_normal_threshold=1.0,
        edge_depth_threshold=0.03,
        sky_mask=sky_mask,
        use_gs_depth=("gs_depth" in predictions),
    )

    imgs_np = imgs[0].permute(0, 2, 3, 1).detach().cpu().numpy()
    depth_np = predictions["depth"][0].detach().cpu().float().numpy()
    normal_np = predictions["normals"][0].detach().cpu().float().numpy()
    cam_poses_np = predictions["camera_poses"][0].detach().cpu().float().numpy()
    cam_intrs_np = predictions["camera_intrs"][0].detach().cpu().float().numpy()

    pts3d_np = depth_to_world_coords_points(
        predictions["depth"][0, ..., 0],
        predictions["camera_poses"][0],
        predictions["camera_intrs"][0],
    )[0].detach().cpu().float().numpy()

    outputs = {
        "images": imgs_np,
        "world_points": pts3d_np,
        "depth": depth_np,
        "normal": normal_np,
        "final_mask": filter_mask,
        "gs_filter_mask": gs_filter_mask,
        "sky_mask": sky_mask,
        "camera_poses": cam_poses_np,
        "camera_intrs": cam_intrs_np,
    }
    if "splats" in predictions:
        sp = predictions["splats"]
        outputs["splats"] = {
            k: sp[k] for k in ("means", "scales", "quats", "opacities")
        }
        for optional in ("sh", "colors", "weights"):
            if optional in sp:
                outputs["splats"][optional] = sp[optional]

    processed_data = _build_vis_data(outputs, imgs)
    torch.cuda.empty_cache()
    return outputs, processed_data


def _build_vis_data(outputs, imgs_tensor):
    """Build per-view visualization dicts from model outputs."""
    vis = {}
    S = outputs["images"].shape[0]
    for i in range(S):
        vis[i] = {
            "image": imgs_tensor[0, i].detach().cpu().numpy(),
            "points3d": outputs["world_points"][i],
            "depth": outputs["depth"][i].squeeze(),
            "normal": outputs["normal"][i],
            "mask": outputs["final_mask"][i].copy(),
        }
    return vis


# ---------------------------------------------------------------------------
# File upload helpers
# ---------------------------------------------------------------------------
def process_uploaded_files(files, time_interval=1.0):
    """Process uploaded files (images or videos) into a target_dir/images folder."""
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"gradio_demo_output/input_images_{timestamp}"
    images_dir = os.path.join(target_dir, "images")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(images_dir)

    image_paths = []
    if files is None:
        return target_dir, image_paths

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".3gp"}

    for file_data in files:
        src_path = str(file_data["name"] if isinstance(file_data, dict) and "name" in file_data else file_data)
        ext = os.path.splitext(src_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        if ext in video_exts:
            cap = cv2.VideoCapture(src_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = max(1, int(fps * time_interval))
            idx, saved = 0, 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if idx % interval == 0:
                    dst = os.path.join(images_dir, f"{base_name}_{saved:06d}.png")
                    cv2.imwrite(dst, frame)
                    image_paths.append(dst)
                    saved += 1
            cap.release()
            print(f"Extracted {saved} frames from {os.path.basename(src_path)}")
        elif ext in {".heic", ".heif"}:
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    dst = os.path.join(images_dir, f"{base_name}.jpg")
                    img.save(dst, "JPEG", quality=95)
                    image_paths.append(dst)
            except Exception:
                dst = os.path.join(images_dir, os.path.basename(src_path))
                shutil.copy(src_path, dst)
                image_paths.append(dst)
        else:
            dst = os.path.join(images_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst)
            image_paths.append(dst)

    image_paths = sorted(image_paths)
    print(f"Processed {len(image_paths)} files to {images_dir}")
    return target_dir, image_paths


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def render_depth_visualization(depth_map, mask=None):
    if depth_map is None:
        return None
    import matplotlib.pyplot as plt
    d = depth_map.copy()
    valid = d > 0
    if mask is not None:
        valid = valid & mask
    if valid.sum() > 0:
        lo, hi = np.percentile(d[valid], 5), np.percentile(d[valid], 95)
        d[valid] = (d[valid] - lo) / (hi - lo + 1e-9)
    rgb = (plt.cm.turbo_r(d)[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = [255, 255, 255]
    return rgb


def render_normal_visualization(normal_map, mask=None):
    if normal_map is None:
        return None
    n = normal_map.copy()
    if mask is not None:
        n[~mask] = 0
    return ((n + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)


def update_view_info(current, total, view_type="Depth"):
    return (
        f"<div style='text-align:center;padding:10px;background:#f8f8f8;color:#999;"
        f"border-radius:8px;margin-bottom:10px;'>"
        f"<strong>{view_type} View Navigation</strong> | "
        f"Current: View {current} / {total} views</div>"
    )


def _get_view(processed_data, idx):
    if not processed_data:
        return None
    keys = list(processed_data.keys())
    idx = max(0, min(idx, len(keys) - 1))
    return processed_data[keys[idx]]


def update_depth_view(pd, idx):
    v = _get_view(pd, idx)
    return render_depth_visualization(v["depth"], v.get("mask")) if v else None


def update_normal_view(pd, idx):
    v = _get_view(pd, idx)
    return render_normal_visualization(v["normal"], v.get("mask")) if v else None


def update_view_selectors(pd):
    n = max(1, len(pd) if pd else 1)
    return (
        gr.update(minimum=1, maximum=n, value=1, step=1),
        gr.update(minimum=1, maximum=n, value=1, step=1),
        update_view_info(1, n, "Depth"),
        update_view_info(1, n, "Normal"),
    )


# ---------------------------------------------------------------------------
# Main reconstruction entry point
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def gradio_demo(
    target_dir,
    frame_selector="All",
    show_camera=False,
    filter_sky_bg=False,
    show_mesh=False,
    filter_ambiguous=False,
):
    tee = TeeOutput()
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        if not target_dir or not os.path.isdir(target_dir):
            sys.stdout = old_stdout
            empty = [None] * 14
            empty[1] = "No valid directory. Please upload files first."
            empty[-1] = tee.getvalue()
            return tuple(empty)

        t0 = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        all_files = sorted(os.listdir(os.path.join(target_dir, "images"))) if os.path.isdir(os.path.join(target_dir, "images")) else []
        frame_choices = ["All"] + [f"{i}: {f}" for i, f in enumerate(all_files)]

        print("Running WorldMirror model...")
        with torch.no_grad():
            predictions, processed_data = run_model(target_dir)

        np.savez(os.path.join(target_dir, "predictions.npz"), **{
            k: v for k, v in predictions.items() if k != "splats"
        })

        save_camera_params(predictions["camera_poses"], predictions["camera_intrs"], target_dir)
        cam_file = os.path.join(target_dir, "camera_params.json")

        if frame_selector is None:
            frame_selector = "All"
        safe = frame_selector.replace(".", "_").replace(":", "").replace(" ", "_")
        glb_path = os.path.join(target_dir, f"scene_{safe}_cam{show_camera}_mesh{show_mesh}_edges{filter_ambiguous}_sky{filter_sky_bg}.glb")
        glb_scene = convert_predictions_to_glb_scene(
            predictions,
            filter_by_frames=frame_selector,
            show_camera=show_camera,
            mask_sky_bg=filter_sky_bg,
            as_mesh=show_mesh,
            mask_ambiguous=filter_ambiguous,
        )
        glb_scene.export(file_obj=glb_path)

        gs_file = None
        if "splats" in predictions:
            gs_file = os.path.join(target_dir, "gaussians.ply")
            save_gaussian_splats_artifact(
                Path(gs_file),
                predictions["splats"],
                filter_mask=predictions.get("final_mask"),
                gs_filter_mask=predictions.get("gs_filter_mask"),
                max_points=5_000_000,
                image_shape=predictions["world_points"].shape[:3],
                memory_budget_gb=4.0,
            )
            print(f"[Save] gaussians.ply ({os.path.getsize(gs_file)} bytes)")

        depth_vis = update_depth_view(processed_data, 0)
        normal_vis = update_normal_view(processed_data, 0)
        d_sl, n_sl, d_info, n_info = update_view_selectors(processed_data)

        del predictions
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        log_msg = f"Reconstruction Success ({len(all_files)} frames, {elapsed:.1f}s)."
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout

        return (
            glb_path,
            log_msg,
            gr.Dropdown(choices=frame_choices, value=frame_selector, interactive=True),
            processed_data,
            depth_vis,
            normal_vis,
            d_sl,
            n_sl,
            d_info,
            n_info,
            cam_file,
            gs_file,
            terminal_log,
        )

    except ValueError as e:
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout
        print(f"[Pipeline] {e}")
        empty = [None] * 13
        empty[1] = f"**Error:** {e}"
        empty[-1] = terminal_log
        return tuple(empty)

    except Exception as e:
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout
        print(f"Error: {e}")
        raise


# ---------------------------------------------------------------------------
# Refresh helpers (interactive parameter changes)
# ---------------------------------------------------------------------------
def refresh_3d_scene(workspace, frame_sel, show_cam, is_example,
                     filter_sky, show_mesh, filter_amb):
    if is_example == "True":
        return gr.update(), gr.update(), "Click Reconstruct first."
    if not workspace or workspace == "None" or not os.path.isdir(workspace):
        return gr.update(), gr.update(), "No results. Please reconstruct first."
    pred_file = os.path.join(workspace, "predictions.npz")
    if not os.path.exists(pred_file):
        return gr.update(), gr.update(), "Missing predictions. Reconstruct first."

    data = np.load(pred_file, allow_pickle=True)
    preds = {k: data[k] for k in data.keys()}

    safe = frame_sel.replace(".", "_").replace(":", "").replace(" ", "_")
    glb_name = f"scene_{safe}_cam{show_cam}_mesh{show_mesh}_edges{filter_amb}_sky{filter_sky}.glb"
    glb_path = os.path.join(workspace, glb_name)
    if not os.path.exists(glb_path):
        scene = convert_predictions_to_glb_scene(
            preds, filter_by_frames=frame_sel, show_camera=show_cam,
            mask_sky_bg=filter_sky, as_mesh=show_mesh, mask_ambiguous=filter_amb,
        )
        scene.export(file_obj=glb_path)

    gs_path = os.path.join(workspace, "gaussians.ply")
    return glb_path, gs_path if os.path.exists(gs_path) else None, "3D scene updated."


def refresh_views_on_filter(workspace, sky_filter, pd, d_slider, n_slider):
    if not workspace or workspace == "None" or not os.path.isdir(workspace):
        return pd, None, None
    pred_file = os.path.join(workspace, "predictions.npz")
    if not os.path.exists(pred_file):
        return pd, None, None
    try:
        data = np.load(pred_file, allow_pickle=True)
        preds = {k: data[k] for k in data.keys()}
        images_dir = os.path.join(workspace, "images")
        img_paths = sorted(
            glob(os.path.join(images_dir, "*.png"))
            + glob(os.path.join(images_dir, "*.jpg"))
            + glob(os.path.join(images_dir, "*.jpeg"))
        )
        effective = compute_adaptive_target_size(img_paths, max_target_size=952)
        imgs = prepare_images_to_tensor(img_paths, target_size=effective).detach().cpu().numpy()

        refreshed = {}
        for i in range(imgs.shape[1]):
            mask = preds["final_mask"][i].copy()
            if sky_filter:
                mask = mask & preds["sky_mask"][i]
            refreshed[i] = {
                "image": imgs[0, i],
                "points3d": preds["world_points"][i],
                "depth": preds["depth"][i].squeeze(),
                "normal": preds["normal"][i],
                "mask": mask,
            }
        di = max(0, int(d_slider or 1) - 1)
        ni = max(0, int(n_slider or 1) - 1)
        return refreshed, update_depth_view(refreshed, di), update_normal_view(refreshed, ni)
    except Exception as e:
        print(f"Error refreshing views: {e}")
        return pd, None, None


# ---------------------------------------------------------------------------
# Example scene helpers
# ---------------------------------------------------------------------------
def extract_example_scenes(base_dir):
    if not os.path.exists(base_dir):
        return []
    exts = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}
    scenes = []
    for name in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, name)
        if not os.path.isdir(d):
            continue
        imgs = []
        for e in exts:
            imgs.extend(glob(os.path.join(d, f"*.{e}")))
            imgs.extend(glob(os.path.join(d, f"*.{e.upper()}")))
        if imgs:
            imgs.sort()
            scenes.append({"name": name, "path": d, "thumbnail": imgs[0],
                           "num_images": len(imgs), "image_files": imgs})
    return scenes


def load_example_scene(scene_name, scenes):
    cfg = next((s for s in scenes if s["name"] == scene_name), None)
    if cfg is None:
        return None, None, None, None, "Scene not found"
    td, paths = process_uploaded_files(cfg["image_files"], 1.0)
    return (None, None, td, paths,
            f"Loaded '{scene_name}' ({cfg['num_images']} images). Click Reconstruct.")


# ---------------------------------------------------------------------------
# Trivial callbacks
# ---------------------------------------------------------------------------
def clear_fields():
    return None

def update_log():
    return "Loading and Reconstructing..."

def get_terminal_output():
    return current_terminal_output

def _safe_int(v, default=1):
    try:
        return int(float(v)) if v is not None else default
    except (ValueError, TypeError):
        return default

def _nav_depth(pd, target):
    if not pd:
        return None, update_view_info(1, 1)
    n = len(pd)
    idx = max(0, min(_safe_int(target) - 1, n - 1))
    return update_depth_view(pd, idx), update_view_info(idx + 1, n)

def _nav_normal(pd, target):
    if not pd:
        return None, update_view_info(1, 1, "Normal")
    n = len(pd)
    idx = max(0, min(_safe_int(target) - 1, n - 1))
    return update_normal_view(pd, idx), update_view_info(idx + 1, n, "Normal")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_demo(examples_dir="./examples/worldrecon"):
    theme = gr.themes.Base()

    css = """
    .custom-log * {
        font-style: italic; font-size: 22px !important;
        background-image: linear-gradient(120deg, #a9b8f8 0%, #7081e8 60%, #4254c5 100%);
        -webkit-background-clip: text; background-clip: text;
        font-weight: bold !important; color: transparent !important; text-align: center !important;
    }
    .normal-weight-btn button, .normal-weight-btn button span { font-weight: 400 !important; }
    .terminal-output { max-height: 400px !important; overflow-y: auto !important; }
    .terminal-output textarea {
        font-family: 'Monaco','Menlo','Ubuntu Mono',monospace !important;
        font-size: 13px !important; line-height: 1.5 !important;
        color: #333 !important; background-color: #f8f9fa !important;
        max-height: 400px !important;
    }
    .example-gallery img { width: 100% !important; height: 280px !important; object-fit: contain !important; }
    .example-col, .example-col > div {
        max-height: 85vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    .example-col .gallery {
        max-height: none !important;
        overflow: visible !important;
    }
    """

    with gr.Blocks(theme=theme, css=css) as demo:
        is_example = gr.Textbox(visible=False, value="None")
        processed_data_state = gr.State(value=None)

        gr.HTML("""
        <div style="text-align:center;">
        <h1>
            <span style="background:linear-gradient(90deg,#3b82f6,#1e40af);-webkit-background-clip:text;
            background-clip:text;color:transparent;font-weight:bold;">WorldMirror 2.0:</span>
            <span style="color:#555;">Universal 3D World Reconstruction</span>
        </h1>
        <p>
        <a href="https://arxiv.org/abs/2510.10726">📄 Paper</a> |
        <a href="https://3d-models.hunyuan.tencent.com/world/">🌐 Project</a> |
        <a href="https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror">💻 GitHub</a> |
        <a href="https://huggingface.co/tencent/HY-World-2.0">🤗 Model</a>
        </p>
        </div>
        <div style="font-size:16px;line-height:1.5;">
        <p>WorldMirror supports any combination of inputs and outputs including point clouds, camera parameters,
        depth maps, normal maps, and 3D Gaussian Splatting.</p>
        <ol>
            <li><strong>Upload</strong> video or images</li>
            <li><strong>Reconstruct</strong> to start 3D processing</li>
            <li><strong>Visualize</strong> across multiple tabs: 3DGS, Point Cloud, Depth, Normal, Camera</li>
        </ol>
        </div>
        """)

        output_path_state = gr.Textbox(visible=False, value="None")

        with gr.Row(equal_height=False):
            # ---------- Left column: Upload ----------
            with gr.Column(scale=1):
                file_upload = gr.File(file_count="multiple", label="Upload Video or Images",
                                      interactive=True, file_types=["image", "video"], height="200px")
                time_interval = gr.Slider(0.1, 10.0, value=1.0, step=0.1,
                                          label="Video Sample Interval (s)", scale=4)
                resample_btn = gr.Button("Resample", scale=1, elem_classes=["normal-weight-btn"])
                image_gallery = gr.Gallery(label="Image Preview", columns=4, height="200px",
                                           show_download_button=True, object_fit="contain", preview=True)
                terminal_output = gr.Textbox(label="Terminal Output", lines=6, max_lines=6,
                                             interactive=False, show_copy_button=True,
                                             elem_classes=["terminal-output"], autoscroll=True)

            # ---------- Center column: Visualization ----------
            with gr.Column(scale=3):
                log_output = gr.Markdown("Upload files, then click Reconstruct.", elem_classes=["custom-log"])

                with gr.Tabs():
                    with gr.Tab("3D Gaussian Splatting", id=1):
                        gs_output = gr.Model3D(label="Gaussian Splatting", height=500)

                    with gr.Tab("Point Cloud / Mesh", id=0):
                        reconstruction_output = gr.Model3D(label="3D Pointmap / Mesh", height=500,
                                                            zoom_speed=0.4, pan_speed=0.4)

                    with gr.Tab("Depth"):
                        depth_view_info = gr.HTML(update_view_info(1, 1))
                        depth_view_slider = gr.Slider(1, 1, step=1, value=1, label="View")
                        depth_map = gr.Image(type="numpy", label="Depth Map", format="png",
                                             interactive=False, height=340)

                    with gr.Tab("Normal"):
                        normal_view_info = gr.HTML(update_view_info(1, 1, "Normal"))
                        normal_view_slider = gr.Slider(1, 1, step=1, value=1, label="View")
                        normal_map = gr.Image(type="numpy", label="Normal Map", format="png",
                                              interactive=False, height=340)

                    with gr.Tab("Camera Parameters"):
                        with gr.Row():
                            gr.HTML("")
                            camera_params_btn = gr.DownloadButton(label="Download Camera Parameters",
                                                                   scale=1, variant="primary")
                            gr.HTML("")

                with gr.Row():
                    reconstruct_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                    clear_btn = gr.ClearButton([
                        file_upload, reconstruction_output, log_output, output_path_state,
                        image_gallery, depth_map, normal_map, depth_view_slider,
                        normal_view_slider, depth_view_info, normal_view_info,
                        camera_params_btn, gs_output,
                    ], scale=1)

                with gr.Row():
                    frame_selector = gr.Dropdown(choices=["All"], value="All",
                                                  label="Show Points of a Specific Frame")

                gr.Markdown("### Reconstruction Options (not applied to 3DGS)")
                with gr.Row():
                    show_camera = gr.Checkbox(label="Show Camera", value=True)
                    show_mesh = gr.Checkbox(label="Show Mesh", value=True)
                    filter_ambiguous = gr.Checkbox(label="Filter low confidence & edges", value=True)
                    filter_sky_bg = gr.Checkbox(label="Filter Sky Background", value=False)

            # ---------- Right column: Examples ----------
            with gr.Column(scale=1, elem_classes=["example-col"]):
                gr.Markdown("### Click to load example scenes")
                real_dir = os.path.join(examples_dir, "realistic")
                style_dir = os.path.join(examples_dir, "stylistic")
                realworld = extract_example_scenes(real_dir) if os.path.isdir(real_dir) else extract_example_scenes(examples_dir)
                stylistic = extract_example_scenes(style_dir) if os.path.isdir(style_dir) else []

                example_outputs = [reconstruction_output, gs_output, output_path_state, image_gallery, log_output]

                def _make_gallery_handler(scene_list):
                    def _handler(evt: gr.SelectData):
                        return load_example_scene(scene_list[evt.index]["name"], scene_list)
                    return _handler

                if not os.path.isdir(real_dir) and not os.path.isdir(style_dir):
                    if realworld:
                        items = [(s["thumbnail"], f"{s['name']}\n{s['num_images']} imgs") for s in realworld]
                        eg = gr.Gallery(value=items, columns=1, height=None, object_fit="contain",
                                        show_label=False, interactive=True, preview=False,
                                        allow_preview=False, elem_classes=["example-gallery"])
                        eg.select(fn=_make_gallery_handler(realworld), outputs=example_outputs)
                else:
                    with gr.Tabs():
                        with gr.Tab("Realistic"):
                            if realworld:
                                items = [(s["thumbnail"], f"{s['name']}\n{s['num_images']} imgs") for s in realworld]
                                rg = gr.Gallery(value=items, columns=1, height=None, object_fit="contain",
                                                show_label=False, interactive=True, preview=False,
                                                allow_preview=False, elem_classes=["example-gallery"])
                                rg.select(fn=_make_gallery_handler(realworld), outputs=example_outputs)
                            else:
                                gr.Markdown("No examples available")
                        with gr.Tab("Stylistic"):
                            if stylistic:
                                items = [(s["thumbnail"], f"{s['name']}\n{s['num_images']} imgs") for s in stylistic]
                                sg = gr.Gallery(value=items, columns=1, height=None, object_fit="contain",
                                                show_label=False, interactive=True, preview=False,
                                                allow_preview=False, elem_classes=["example-gallery"])
                                sg.select(fn=_make_gallery_handler(stylistic), outputs=example_outputs)
                            else:
                                gr.Markdown("No examples available")

        # ---- Events: Reconstruct ----
        reconstruct_btn.click(clear_fields, [], []).then(
            update_log, [], [log_output]
        ).then(
            gradio_demo,
            [output_path_state, frame_selector, show_camera, filter_sky_bg, show_mesh, filter_ambiguous],
            [reconstruction_output, log_output, frame_selector, processed_data_state,
             depth_map, normal_map, depth_view_slider, normal_view_slider,
             depth_view_info, normal_view_info, camera_params_btn, gs_output, terminal_output],
        ).then(lambda: "False", [], [is_example])

        # ---- Events: Refresh 3D scene on option change ----
        _scene_inputs = [output_path_state, frame_selector, show_camera, is_example,
                         filter_sky_bg, show_mesh, filter_ambiguous]
        _scene_outputs = [reconstruction_output, gs_output, log_output]
        for ctrl in (frame_selector, show_camera, show_mesh):
            ctrl.change(refresh_3d_scene, _scene_inputs, _scene_outputs)

        _filter_inputs = [output_path_state, filter_sky_bg, processed_data_state,
                          depth_view_slider, normal_view_slider]
        _filter_outputs = [processed_data_state, depth_map, normal_map]

        for ctrl in (filter_sky_bg, filter_ambiguous):
            ctrl.change(refresh_3d_scene, _scene_inputs, _scene_outputs).then(
                refresh_views_on_filter, _filter_inputs, _filter_outputs)

        # ---- Events: File upload ----
        def _on_upload(files, interval):
            if not files:
                return None, None, None, ""
            tee = TeeOutput()
            old = sys.stdout
            sys.stdout = tee
            try:
                td, paths = process_uploaded_files(files, interval)
                log = tee.getvalue()
                sys.stdout = old
                return td, paths, "Upload complete. Click Reconstruct.", log
            except Exception:
                sys.stdout = old
                raise

        def _on_resample(files, interval, cur_dir):
            if not files:
                return cur_dir, None, "No files.", ""
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
            has_video = any(
                os.path.splitext(str(f["name"] if isinstance(f, dict) else f))[1].lower() in video_exts
                for f in files
            )
            if not has_video:
                return cur_dir, None, "No video found.", ""
            if cur_dir and cur_dir != "None" and os.path.exists(cur_dir):
                shutil.rmtree(cur_dir)
            tee = TeeOutput()
            old = sys.stdout
            sys.stdout = tee
            try:
                td, paths = process_uploaded_files(files, interval)
                log = tee.getvalue()
                sys.stdout = old
                return td, paths, f"Resampled ({interval}s interval). Click Reconstruct.", log
            except Exception:
                sys.stdout = old
                raise

        file_upload.change(_on_upload, [file_upload, time_interval],
                           [output_path_state, image_gallery, log_output, terminal_output])
        resample_btn.click(_on_resample, [file_upload, time_interval, output_path_state],
                            [output_path_state, image_gallery, log_output, terminal_output])

        # ---- Events: Depth / Normal navigation ----
        depth_view_slider.change(_nav_depth, [processed_data_state, depth_view_slider],
                                  [depth_map, depth_view_info])
        normal_view_slider.change(_nav_normal, [processed_data_state, normal_view_slider],
                                   [normal_map, normal_view_info])

        # ---- Terminal polling ----
        timer = gr.Timer(value=0.5)
        timer.tick(get_terminal_output, [], [terminal_output])

        gr.HTML("""
        <hr style="margin-top:40px;margin-bottom:20px;">
        <div style="text-align:center;font-size:14px;color:#666;margin-bottom:20px;">
            <h3>Acknowledgements</h3>
            <p>🔗 <a href="https://github.com/facebookresearch/vggt">VGGT</a> |
            🔗 <a href="https://github.com/CUT3R/CUT3R">CUT3R</a></p>
        </div>
        """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WorldMirror 2.0 Gradio Demo")
    parser.add_argument("--examples_dir", type=str, default="./examples/worldrecon",
                        help="Path to example scenes directory")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Training config YAML (used with --ckpt_path)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="tencent/HY-World-2.0",
                        help="HuggingFace repo ID or local path")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Local checkpoint file (.ckpt / .safetensors)")
    parser.add_argument("--use_fsdp", action="store_true", default=False,
                        help="Enable FSDP multi-GPU sharding")
    parser.add_argument("--enable_bf16", action="store_true", default=False,
                        help="Enable bfloat16 mixed precision")
    parser.add_argument("--fsdp_cpu_offload", action="store_true", default=False,
                        help="Offload FSDP params to CPU")
    args = parser.parse_args()

    _init_pipeline_args(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
        use_fsdp=args.use_fsdp,
        enable_bf16=args.enable_bf16,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
    )

    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if is_distributed:
        pipe = _get_pipeline()
        rank = pipe.rank

        if rank == 0:
            print("[Gradio] Rank 0: launching Gradio server")
            demo = build_demo(examples_dir=args.examples_dir)
            try:
                demo.queue().launch(
                    show_error=True,
                    share=args.share,
                    server_name=args.host,
                    server_port=args.port,
                    ssr_mode=False,
                )
            finally:
                _broadcast_string("__QUIT__", rank, src=0)
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
        else:
            _worker_loop()
    else:
        demo = build_demo(examples_dir=args.examples_dir)
        demo.queue().launch(
            show_error=True,
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            ssr_mode=False,
        )
