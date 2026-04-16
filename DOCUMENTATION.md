# HunyuanWorld 2.0 — Documentation
This document provides detailed usage guides, parameter references, and output format specifications for each component of HunyuanWorld 2.0.

## Table of Contents
- [WorldMirror 2.0 (World Reconstruction)](#worldmirror-20-world-reconstruction)
  - [Overview](#overview)
  - [Python API](#python-api)
    - [`WorldMirrorPipeline.from_pretrained`](#worldmirrorpipelinefrom_pretrained)
    - [`WorldMirrorPipeline.__call__`](#worldmirrorpipelinecall)
  - [CLI Reference](#cli-reference)
  - [Output Format](#output-format)
    - [File Structure](#file-structure)
    - [Prediction Dictionary](#prediction-dictionary)
  - [Prior Injection](#prior-injection)
    - [Camera Parameters (JSON)](#camera-parameters-json)
    - [Depth Maps (Folder)](#depth-maps-folder)
    - [Combining Priors](#combining-priors)
  - [Multi-GPU Inference](#multi-gpu-inference)
  - [Advanced Options](#advanced-options)
    - [Disabling Prediction Heads](#disabling-prediction-heads)
    - [Mask Filtering](#mask-filtering)
    - [Point Cloud Compression](#point-cloud-compression)
  - [Gradio App](#gradio-app)
- [Panorama Generation](#panorama-generation)
- [World Generation](#world-generation)

---
## WorldMirror 2.0 (World Reconstruction)
### Overview
WorldMirror 2.0 is a unified feed-forward model for comprehensive 3D geometric prediction from multi-view images or video. It simultaneously generates:
- **3D point clouds** in world coordinates
- **Per-view depth maps** in camera frame
- **Surface normals** in camera coordinates
- **Camera poses** (c2w) and **intrinsics**
- **3D Gaussian Splatting** attributes (means, scales, rotations, opacities, SH coefficients)

Key improvements over WorldMirror 1.0:
- **Normalized RoPE** for flexible resolution inference
- **Depth mask prediction** for robust invalid pixel handling
- **Sequence Parallel + FSDP + BF16** for efficient multi-GPU inference

---
### Python API
#### `WorldMirrorPipeline.from_pretrained`
Factory method to load the model and create a pipeline instance.

```python
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

pipeline = WorldMirrorPipeline.from_pretrained(
    pretrained_model_name_or_path="tencent/HY-World-2.0",
    subfolder="HY-WorldMirror-2.0",
    config_path=None,
    ckpt_path=None,
    use_fsdp=False,
    enable_bf16=False,
    fsdp_cpu_offload=False,
    disable_heads=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_model_name_or_path` | `str` | `"tencent/HY-World-2.0"` | HuggingFace repo ID or local path |
| `subfolder` | `str` | `"HY-WorldMirror-2.0"` | Subfolder inside the repo containing WorldMirror checkpoint (`model.safetensors` + config) |
| `config_path` | `str` | `None` | Training config YAML (used with `ckpt_path` for custom checkpoints) |
| `ckpt_path` | `str` | `None` | Checkpoint file (`.ckpt` / `.safetensors`). When provided with `config_path`, loads model from local checkpoint instead of HuggingFace |
| `use_fsdp` | `bool` | `False` | Shard parameters across GPUs via Fully Sharded Data Parallel |
| `enable_bf16` | `bool` | `False` | Use bfloat16 precision (except numerically critical layers) |
| `fsdp_cpu_offload` | `bool` | `False` | Offload FSDP parameters to CPU (saves GPU memory at the cost of speed) |
| `disable_heads` | `list[str]` | `None` | Heads to disable and free from memory. Options: `"camera"`, `"depth"`, `"normal"`, `"points"`, `"gs"` |

**Notes:**
- Distributed mode is auto-detected from `WORLD_SIZE` environment variable (set by `torchrun`).
- When using multi-GPU, each rank must call `from_pretrained` — the method handles `dist.init_process_group` internally.

---
#### `WorldMirrorPipeline.__call__`
Run inference on a set of images or a video.

```python
result = pipeline(
    input_path,
    output_path="inference_output",
    **kwargs,
)
```

Returns the output directory path (`str`), or `None` if the input was skipped.

**Inference Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | *(required)* | Directory of images or path to a video file |
| `output_path` | `str` | `"inference_output"` | Root output directory |
| `target_size` | `int` | `952` | Maximum inference resolution (longest edge). Images are resized + center-cropped to the nearest multiple of 14 |
| `fps` | `int` | `1` | FPS for extracting frames from video input |
| `video_strategy` | `str` | `"new"` | Video frame extraction strategy: `"new"` (motion-aware) or `"old"` (uniform FPS) |
| `video_min_frames` | `int` | `1` | Minimum number of frames to extract from video |
| `video_max_frames` | `int` | `32` | Maximum number of frames to extract from video |

**Save Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_depth` | `bool` | `True` | Save per-view depth maps (PNG visualization + NPY raw values) |
| `save_normal` | `bool` | `True` | Save per-view surface normal maps (PNG) |
| `save_gs` | `bool` | `True` | Save 3D Gaussian Splatting as `gaussians.ply` |
| `save_camera` | `bool` | `True` | Save camera parameters as `camera_params.json` |
| `save_points` | `bool` | `True` | Save depth-derived point cloud as `points.ply` |
| `save_colmap` | `bool` | `False` | Save COLMAP-format sparse reconstruction (`sparse/0/`) |
| `save_conf` | `bool` | `False` | Save depth confidence maps |
| `save_sky_mask` | `bool` | `False` | Save sky segmentation masks |

**Mask Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_sky_mask` | `bool` | `True` | Filter out sky regions from point clouds and Gaussians |
| `apply_edge_mask` | `bool` | `True` | Filter out edge/discontinuity regions |
| `apply_confidence_mask` | `bool` | `False` | Filter out low-confidence predictions |
| `sky_mask_source` | `str` | `"auto"` | Sky mask method: `"auto"` (ONNX + model fusion), `"model"` (model predictions only), `"onnx"` (external segmentation only) |
| `model_sky_threshold` | `float` | `0.45` | Threshold for model-based sky detection |
| `confidence_percentile` | `float` | `10.0` | Percentile threshold for confidence filtering (bottom N% removed) |
| `edge_normal_threshold` | `float` | `1.0` | Normal edge detection tolerance |
| `edge_depth_threshold` | `float` | `0.03` | Depth edge detection relative tolerance |

**Compression Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compress_pts` | `bool` | `True` | Compress point clouds via voxel merging + random sampling |
| `compress_pts_max_points` | `int` | `2,000,000` | Maximum number of points after compression |
| `compress_pts_voxel_size` | `float` | `0.002` | Voxel size for point cloud merging |
| `max_resolution` | `int` | `1920` | Maximum resolution for saved output images |
| `compress_gs_max_points` | `int` | `5,000,000` | Maximum number of Gaussians after voxel pruning |
| `output_memory_budget_gb` | `float` | `4.0` | Approximate host-memory budget for chunked output serialization |
| `save_chunk_frames` | `int` | `None` | Optional fixed number of frames per output chunk; overrides automatic chunk sizing |

**Prior Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior_cam_path` | `str` | `None` | Path to camera parameters JSON file |
| `prior_depth_path` | `str` | `None` | Path to directory containing depth map files |

**Rendered Video Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_rendered` | `bool` | `False` | Render interpolated fly-through video from Gaussian splats |
| `render_interp_per_pair` | `int` | `15` | Number of interpolated frames between each camera pair |
| `render_depth` | `bool` | `False` | Also render a depth visualization video |

**Misc Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_time` | `bool` | `True` | Print timing report and save `pipeline_timing.json` |
| `strict_output_path` | `str` | `None` | If set, save results directly to this path without `<case_name>/<timestamp>` subdirectories |

---
### CLI Reference
All `__call__` parameters are exposed as CLI arguments:

```bash
python -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --output_path inference_output \
    --target_size 952 \
    --prior_cam_path path/to/camera_params.json \
    --prior_depth_path path/to/depth_dir/ \
```

**Boolean flag conventions:**

| Enable | Disable |
|--------|---------|
| `--save_colmap` | *(omit)* |
| `--save_conf` | *(omit)* |
| `--save_sky_mask` | *(omit)* |
| `--apply_sky_mask` (default on) | `--no_sky_mask` |
| `--apply_edge_mask` (default on) | `--no_edge_mask` |
| `--apply_confidence_mask` | *(omit)* |
| `--compress_pts` (default on) | `--no_compress_pts` |
| `--log_time` (default on) | `--no_log_time` |
| *(default on)* `save_depth` | `--no_save_depth` |
| *(default on)* `save_normal` | `--no_save_normal` |
| *(default on)* `save_gs` | `--no_save_gs` |
| *(default on)* `save_camera` | `--no_save_camera` |
| *(default on)* `save_points` | `--no_save_points` |
| `--save_rendered` | *(omit)* |
| `--render_depth` | *(omit)* |

**Additional CLI-only arguments:**

| Argument | Description |
|----------|-------------|
| `--config_path` | Training config YAML for custom checkpoint loading |
| `--ckpt_path` | Local checkpoint file path |
| `--use_fsdp` | Enable FSDP multi-GPU sharding |
| `--enable_bf16` | Enable bfloat16 mixed precision |
| `--fsdp_cpu_offload` | Offload FSDP params to CPU |
| `--disable_heads` | Space-separated list of heads to disable (e.g. `--disable_heads camera normal`) |
| `--output_memory_budget_gb` | Host-memory budget used to size output serialization chunks |
| `--save_chunk_frames` | Fixed frame count per save chunk, useful for deterministic memory envelopes |
| `--no_interactive` | Exit after first inference (skip interactive prompt loop) |

---
### Output Format
#### File Structure

```
inference_output/
└── <case_name>/
    └── <timestamp>/
        ├── depth/
        │   ├── depth_0000.png      # Normalized depth visualization
        │   ├── depth_0000.npy      # Raw float32 depth values [H, W]
        │   └── ...
        ├── normal/
        │   ├── normal_0000.png     # Normal map visualization (RGB)
        │   └── ...
        ├── camera_params.json      # Camera extrinsics & intrinsics
        ├── gaussians.ply           # 3D Gaussian Splatting (standard format)
        ├── points.ply              # Colored point cloud
        ├── sparse/                 # COLMAP format (if --save_colmap)
        │   └── 0/
        │       ├── cameras.bin
        │       ├── images.bin
        │       └── points3D.bin
        ├── rendered/               # Rendered video (if --save_rendered)
        │   ├── rendered_rgb.mp4
        │   └── rendered_depth.mp4  # (if --render_depth)
        └── pipeline_timing.json    # Performance timing report
```

#### Prediction Dictionary
When using the Python API, `pipeline(...)` internally produces a `predictions` dictionary with the following keys:

```python
# Geometry
predictions["depth"]        # [B, S, H, W, 1]  — Z-depth in camera frame
predictions["depth_conf"]   # [B, S, H, W]     — Depth confidence
predictions["normals"]      # [B, S, H, W, 3]  — Surface normals in camera coords
predictions["normals_conf"] # [B, S, H, W]     — Normal confidence
predictions["pts3d"]        # [B, S, H, W, 3]  — 3D point maps in world coords
predictions["pts3d_conf"]   # [B, S, H, W]     — Point cloud confidence
# Camera
predictions["camera_poses"] # [B, S, 4, 4]     — Camera-to-world (c2w), OpenCV convention
predictions["camera_intrs"] # [B, S, 3, 3]     — Camera intrinsic matrices
predictions["camera_params"]# [B, S, 9]        — Compact camera vector (translation, quaternion, fov_v, fov_u)
# 3D Gaussian Splatting
predictions["splats"]["means"]      # [B, N, 3] — Gaussian centers
predictions["splats"]["scales"]     # [B, N, 3] — Gaussian scales
predictions["splats"]["quats"]      # [B, N, 4] — Gaussian rotations (quaternions)
predictions["splats"]["opacities"]  # [B, N]    — Gaussian opacities
predictions["splats"]["sh"]         # [B, N, 1, 3] — Spherical harmonics (degree 0)
predictions["splats"]["weights"]    # [B, N]    — Per-Gaussian confidence weights
```

Where `B` = batch size (always 1 for inference), `S` = number of input views, `H, W` = image dimensions, `N` = total Gaussians (`S × H × W`).

---
### Prior Injection
WorldMirror 2.0 accepts three types of geometric priors as conditioning inputs. Priors are automatically detected from the provided files.

| Prior Type | Condition | Input Format |
|------------|-----------|--------------|
| Camera Pose | `cond_flags[0]` | c2w 4×4 matrix (OpenCV convention) |
| Depth Map | `cond_flags[1]` | Per-view float depth maps |
| Intrinsics | `cond_flags[2]` | 3×3 intrinsic matrix |

#### Camera Parameters (JSON)
The camera parameter file follows the same format as the `camera_params.json` output by the pipeline:

```json
{
  "num_cameras": 2,
  "extrinsics": [
    {
      "camera_id": 0,
      "matrix": [
        [0.98, 0.01, -0.17, 0.52],
        [-0.01, 0.99, 0.01, -0.03],
        [0.17, -0.01, 0.98, 1.20],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ],
  "intrinsics": [
    {
      "camera_id": 0,
      "matrix": [
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `camera_id` | Integer index (`0`, `1`, `2`, ...) or image filename stem without extension (e.g., `"image_0001"`) |
| `extrinsics.matrix` | 4×4 camera-to-world (c2w) transformation matrix, OpenCV coordinate convention |
| `intrinsics.matrix` | 3×3 camera intrinsic matrix in pixels (`fx, fy` = focal lengths; `cx, cy` = principal point) |

**Important notes:**
- `extrinsics` and `intrinsics` lists can be provided independently or together. An empty list `[]` or missing key means that prior is unavailable.
- **Intrinsics resolution:** Values should correspond to the **original image resolution**. The pipeline automatically adjusts for inference-time resize + center-crop.
- **Extrinsics alignment:** The pipeline automatically normalizes all extrinsics relative to the first view, consistent with training behavior.
#### Depth Maps (Folder)
Depth maps are stored as individual files in a directory. Filenames should match the input image filenames. Supported formats: `.npy`, `.exr`, `.png` (16-bit).

```
prior_depth/
├── image_0001.npy    # float32, shape [H, W]
├── image_0002.npy
└── ...
```

#### Combining Priors
Priors can be freely combined. Examples:

```bash
# Only intrinsics
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_cam_path camera_intrinsics_only.json
# Only depth
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_depth_path depth_maps/
# Camera poses + intrinsics + depth
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_cam_path camera_params.json \
    --prior_depth_path depth_maps/
```

---
### Multi-GPU Inference
WorldMirror 2.0 supports **Sequence Parallel (SP)** inference across multiple GPUs, where token sequences are sharded across ranks in the ViT backbone, and DPT heads process frames in parallel.

> **Requirement:** The number of input images must be **>= the number of GPUs** (`nproc_per_node`). For example, with 8 GPUs you need at least 8 input images. The pipeline will raise an error if this condition is not met.

```bash
# 2 GPUs with FSDP + bf16
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --use_fsdp --enable_bf16
# 4 GPUs
torchrun --nproc_per_node=4 -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --use_fsdp --enable_bf16
# Python API (inside a torchrun script)
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
pipeline = WorldMirrorPipeline.from_pretrained(
    'tencent/HY-World-2.0',
    use_fsdp=True,
    enable_bf16=True,
)
pipeline('path/to/images')
```

**What happens under the hood:**
1. `from_pretrained` auto-detects `WORLD_SIZE > 1` and initializes `torch.distributed`.
2. The model is loaded on rank 0 and broadcast via `sync_module_states=True`.
3. FSDP shards parameters across the SP process group.
4. DPT prediction heads split frames across ranks and `AllGather` results.
5. Post-processing (mask computation, saving) runs on rank 0 only. Large 3DGS and point-cloud outputs are serialized in chunks so rank 0 does not need a full CPU copy of every per-pixel Gaussian or point at once.

---
### Advanced Options
#### Disabling Prediction Heads
To save memory when you only need specific outputs:

```python
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

pipeline = WorldMirrorPipeline.from_pretrained(
    'tencent/HY-World-2.0',
    disable_heads=["normal", "points"],  # free ~200M params
)
```

Available heads: `"camera"`, `"depth"`, `"normal"`, `"points"`, `"gs"`.
#### Mask Filtering
The pipeline supports three types of output filtering to improve point cloud and Gaussian quality:
1. **Sky mask** (`apply_sky_mask=True`): Removes sky regions using an ONNX-based segmentation model, optionally fused with model-predicted depth masks.
2. **Edge mask** (`apply_edge_mask=True`): Removes points at depth/normal discontinuities (object boundaries).
3. **Confidence mask** (`apply_confidence_mask=False`): Removes the bottom N% of points by prediction confidence.
These masks are applied independently to both the `points.ply` (depth-based) and `gaussians.ply` (GS-based) outputs. The GS output uses its own depth predictions for edge detection when available.
#### Point Cloud Compression
WorldMirror writes large point-cloud and 3DGS artifacts through a chunked output path. By default, `output_memory_budget_gb=4.0` is used to size transient CPU buffers while preserving the final compression settings. For tighter hosts, lower the budget:

```bash
python -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --output_memory_budget_gb 2.0
```

For repeatable chunk boundaries across machines, set `--save_chunk_frames`:

```bash
python -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --save_chunk_frames 2
```

When `compress_pts=True` (default), the depth-derived point cloud undergoes:
1. **Voxel merging**: Points within each voxel (size controlled by `compress_pts_voxel_size`) are merged via weighted averaging.
2. **Random subsampling**: If the result exceeds `compress_pts_max_points`, points are uniformly subsampled.
Similarly, Gaussians are voxel-pruned (weighted averaging of means, scales, quaternions, colors, opacities) and optionally subsampled to `compress_gs_max_points`.

---
### Gradio App
An interactive web demo for WorldMirror 2.0. Upload images or videos and visualize 3DGS, point clouds, depth maps, normal maps, and camera parameters in your browser.
**Quick start:**

```bash
# Single GPU
python -m hyworld2.worldrecon.gradio_app

# Multi-GPU
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.gradio_app \
    --use_fsdp --enable_bf16
```

**With a local checkpoint:**

```bash
python -m hyworld2.worldrecon.gradio_app \
    --config_path /path/to/config.yaml \
    --ckpt_path /path/to/checkpoint.safetensors
```

**With a public link (e.g., for Colab or remote servers):**

```bash
python -m hyworld2.worldrecon.gradio_app --share
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `8081` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--share` | `False` | Create a public Gradio link |
| `--examples_dir` | `./examples/worldrecon` | Path to example scenes directory |
| `--config_path` | `None` | Training config YAML (used with `--ckpt_path`) |
| `--ckpt_path` | `None` | Local checkpoint file (`.ckpt` / `.safetensors`) |
| `--use_fsdp` | `False` | Enable FSDP multi-GPU sharding |
| `--enable_bf16` | `False` | Enable bfloat16 mixed precision |
| `--fsdp_cpu_offload` | `False` | Offload FSDP params to CPU (saves GPU memory) |

> **Important:** In multi-GPU mode, the number of input images must be **>= the number of GPUs**.

---
## Panorama Generation
*Coming soon.*
This section will document the panorama generation model, including:
- Text-to-panorama and image-to-panorama APIs
- Model architecture (MMDiT-based implicit perspective-to-ERP mapping)
- Configuration parameters
- Output formats

---
## World Generation
*Coming soon.*
This section will document the world generation pipeline, including:
- Trajectory planning configuration
- World expansion with memory-driven video generation
- World composition (point cloud expansion + 3DGS optimization)
- End-to-end generation from text/image to navigable 3D world
