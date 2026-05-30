# HunyuanWorld 2.0 — 文档
本文档提供了 HunyuanWorld 2.0 各组件的详细使用指南、参数参考和输出格式说明。

## 目录
- [WorldMirror 2.0（世界重建）](#worldmirror-20世界重建)
  - [概述](#概述)
  - [Python API](#python-api)
    - [`WorldMirrorPipeline.from_pretrained`](#worldmirrorpipelinefrom_pretrained)
    - [`WorldMirrorPipeline.__call__`](#worldmirrorpipelinecall)
  - [命令行参考](#命令行参考)
  - [输出格式](#输出格式)
    - [文件结构](#文件结构)
    - [预测字典](#预测字典)
  - [先验注入](#先验注入)
    - [相机参数（JSON）](#相机参数json)
    - [深度图（文件夹）](#深度图文件夹)
    - [组合先验](#组合先验)
  - [多卡推理](#多卡推理)
  - [高级选项](#高级选项)
    - [禁用预测头](#禁用预测头)
    - [掩码过滤](#掩码过滤)
    - [点云压缩](#点云压缩)
  - [Gradio 应用](#gradio-应用)
- [全景生成](#全景生成)
- [世界生成](#世界生成)

---
## WorldMirror 2.0（世界重建）
### 概述
WorldMirror 2.0 是一个统一的前馈模型，用于从多视图图像或视频进行全面的3D几何预测。它能同时生成：
- **3D 点云**（世界坐标系）
- **逐视图深度图**（相机坐标系）
- **表面法线**（相机坐标系）
- **相机位姿**（c2w）和**内参**
- **3D 高斯点云**属性（均值、尺度、旋转、不透明度、球谐系数）

相比 WorldMirror 1.0 的关键改进：
- **归一化 RoPE** 支持灵活分辨率推理
- **深度掩码预测** 实现稳健的无效像素处理
- **序列并行 + FSDP + BF16** 实现高效多卡推理

---
### Python API
#### `WorldMirrorPipeline.from_pretrained`
工厂方法，用于加载模型并创建 Pipeline 实例。

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

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `pretrained_model_name_or_path` | `str` | `"tencent/HY-World-2.0"` | HuggingFace 仓库 ID 或本地路径 |
| `subfolder` | `str` | `"HY-WorldMirror-2.0"` | 仓库内包含 WorldMirror 检查点的子文件夹（`model.safetensors` + 配置文件） |
| `config_path` | `str` | `None` | 训练配置 YAML（与 `ckpt_path` 配合用于自定义检查点） |
| `ckpt_path` | `str` | `None` | 检查点文件（`.ckpt` / `.safetensors`）。与 `config_path` 一起使用时，从本地检查点加载模型而非 HuggingFace |
| `use_fsdp` | `bool` | `False` | 通过完全分片数据并行（FSDP）在多卡间分片参数 |
| `enable_bf16` | `bool` | `False` | 使用 bfloat16 精度（数值敏感层除外） |
| `fsdp_cpu_offload` | `bool` | `False` | 将 FSDP 参数卸载到 CPU（节省显存但降低速度） |
| `disable_heads` | `list[str]` | `None` | 要禁用并释放内存的预测头。可选：`"camera"`、`"depth"`、`"normal"`、`"points"`、`"gs"` |

**说明：**
- 分布式模式通过 `WORLD_SIZE` 环境变量（由 `torchrun` 设置）自动检测。
- 使用多卡时，每个 rank 都需要调用 `from_pretrained`——该方法内部处理 `dist.init_process_group`。

---
#### `WorldMirrorPipeline.__call__`
对一组图像或视频运行推理。

```python
result = pipeline(
    input_path,
    output_path="inference_output",
    **kwargs,
)
```

返回输出目录路径（`str`），如果输入被跳过则返回 `None`。

**推理参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_path` | `str` | *（必填）* | 图像目录或视频文件路径 |
| `output_path` | `str` | `"inference_output"` | 输出根目录 |
| `target_size` | `int` | `952` | 最大推理分辨率（最长边）。图像将被缩放 + 中心裁剪到最近的 14 的倍数 |
| `fps` | `int` | `1` | 从视频输入提取帧的帧率 |
| `video_strategy` | `str` | `"new"` | 视频帧提取策略：`"new"`（运动感知）或 `"old"`（均匀 FPS） |
| `video_min_frames` | `int` | `1` | 从视频中提取的最少帧数 |
| `video_max_frames` | `int` | `32` | 从视频中提取的最多帧数 |

**保存参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `save_depth` | `bool` | `True` | 保存逐视图深度图（PNG 可视化 + NPY 原始值） |
| `save_normal` | `bool` | `True` | 保存逐视图表面法线图（PNG） |
| `save_gs` | `bool` | `True` | 保存 3D 高斯点云为 `gaussians.ply` |
| `save_camera` | `bool` | `True` | 保存相机参数为 `camera_params.json` |
| `save_points` | `bool` | `True` | 保存基于深度的点云为 `points.ply` |
| `save_colmap` | `bool` | `False` | 保存 COLMAP 格式的稀疏重建（`sparse/0/`） |
| `save_conf` | `bool` | `False` | 保存深度置信度图 |
| `save_sky_mask` | `bool` | `False` | 保存天空分割掩码 |

**掩码参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `apply_sky_mask` | `bool` | `True` | 从点云和高斯中过滤天空区域 |
| `apply_edge_mask` | `bool` | `True` | 过滤边缘/不连续区域 |
| `apply_confidence_mask` | `bool` | `False` | 过滤低置信度预测 |
| `sky_mask_source` | `str` | `"auto"` | 天空掩码方法：`"auto"`（ONNX + 模型融合）、`"model"`（仅模型预测）、`"onnx"`（仅外部分割） |
| `model_sky_threshold` | `float` | `0.45` | 基于模型的天空检测阈值 |
| `confidence_percentile` | `float` | `10.0` | 置信度过滤的百分位阈值（移除最低 N%） |
| `edge_normal_threshold` | `float` | `1.0` | 法线边缘检测容差 |
| `edge_depth_threshold` | `float` | `0.03` | 深度边缘检测相对容差 |

**压缩参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `compress_pts` | `bool` | `True` | 通过体素合并 + 随机采样压缩点云 |
| `compress_pts_max_points` | `int` | `2,000,000` | 压缩后的最大点数 |
| `compress_pts_voxel_size` | `float` | `0.002` | 点云合并的体素大小 |
| `max_resolution` | `int` | `1920` | 保存输出图像的最大分辨率 |
| `compress_gs_max_points` | `int` | `5,000,000` | 体素剪枝后的最大高斯数 |

**先验参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `prior_cam_path` | `str` | `None` | 相机参数 JSON 文件路径 |
| `prior_depth_path` | `str` | `None` | 深度图文件夹路径 |

**渲染视频参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `save_rendered` | `bool` | `False` | 从高斯点云渲染插值飞行视频 |
| `render_interp_per_pair` | `int` | `15` | 每对相机之间的插值帧数 |
| `render_depth` | `bool` | `False` | 同时渲染深度可视化视频 |

**其他参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `log_time` | `bool` | `True` | 打印计时报告并保存 `pipeline_timing.json` |
| `strict_output_path` | `str` | `None` | 若指定，结果直接保存到该路径下，不创建 `<case_name>/<timestamp>` 子目录 |

---
### 命令行参考
所有 `__call__` 参数都可作为命令行参数使用：

```bash
python -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --output_path inference_output \
    --target_size 952 \
    --prior_cam_path path/to/camera_params.json \
    --prior_depth_path path/to/depth_dir/
```

**布尔标志约定：**

| 启用 | 禁用 |
|------|------|
| `--save_colmap` | *（省略）* |
| `--save_conf` | *（省略）* |
| `--save_sky_mask` | *（省略）* |
| `--apply_sky_mask`（默认开启） | `--no_sky_mask` |
| `--apply_edge_mask`（默认开启） | `--no_edge_mask` |
| `--apply_confidence_mask` | *（省略）* |
| `--compress_pts`（默认开启） | `--no_compress_pts` |
| `--log_time`（默认开启） | `--no_log_time` |
| *（默认开启）* `save_depth` | `--no_save_depth` |
| *（默认开启）* `save_normal` | `--no_save_normal` |
| *（默认开启）* `save_gs` | `--no_save_gs` |
| *（默认开启）* `save_camera` | `--no_save_camera` |
| *（默认开启）* `save_points` | `--no_save_points` |
| `--save_rendered` | *（省略）* |
| `--render_depth` | *（省略）* |

**仅命令行参数：**

| 参数 | 描述 |
|------|------|
| `--config_path` | 用于自定义检查点加载的训练配置 YAML |
| `--ckpt_path` | 本地检查点文件路径 |
| `--use_fsdp` | 启用 FSDP 多卡分片 |
| `--enable_bf16` | 启用 bfloat16 混合精度 |
| `--fsdp_cpu_offload` | 将 FSDP 参数卸载到 CPU |
| `--disable_heads` | 以空格分隔要禁用的预测头（例如 `--disable_heads camera normal`） |
| `--no_interactive` | 首次推理后退出（跳过交互式提示循环） |

---
### 输出格式
#### 文件结构

```
inference_output/
└── <case_name>/
    └── <timestamp>/
        ├── depth/
        │   ├── depth_0000.png      # 归一化深度可视化
        │   ├── depth_0000.npy      # 原始 float32 深度值 [H, W]
        │   └── ...
        ├── normal/
        │   ├── normal_0000.png     # 法线图可视化（RGB）
        │   └── ...
        ├── camera_params.json      # 相机外参和内参
        ├── gaussians.ply           # 3D 高斯点云（标准格式）
        ├── points.ply              # 带颜色的点云
        ├── sparse/                 # COLMAP 格式（使用 --save_colmap 时）
        │   └── 0/
        │       ├── cameras.bin
        │       ├── images.bin
        │       └── points3D.bin
        ├── rendered/               # 渲染视频（使用 --save_rendered 时）
        │   ├── rendered_rgb.mp4
        │   └── rendered_depth.mp4  # （使用 --render_depth 时）
        └── pipeline_timing.json    # 性能计时报告
```

#### 预测字典
使用 Python API 时，`pipeline(...)` 内部生成一个 `predictions` 字典，包含以下键：

```python
# 几何
predictions["depth"]        # [B, S, H, W, 1]  — 相机坐标系中的 Z 深度
predictions["depth_conf"]   # [B, S, H, W]     — 深度置信度
predictions["normals"]      # [B, S, H, W, 3]  — 相机坐标系中的表面法线
predictions["normals_conf"] # [B, S, H, W]     — 法线置信度
predictions["pts3d"]        # [B, S, H, W, 3]  — 世界坐标系中的 3D 点图
predictions["pts3d_conf"]   # [B, S, H, W]     — 点云置信度
# 相机
predictions["camera_poses"] # [B, S, 4, 4]     — 相机到世界（c2w），OpenCV 约定
predictions["camera_intrs"] # [B, S, 3, 3]     — 相机内参矩阵
predictions["camera_params"]# [B, S, 9]        — 紧凑相机向量（平移、四元数、fov_v、fov_u）
# 3D 高斯点云
predictions["splats"]["means"]      # [B, N, 3] — 高斯中心
predictions["splats"]["scales"]     # [B, N, 3] — 高斯尺度
predictions["splats"]["quats"]      # [B, N, 4] — 高斯旋转（四元数）
predictions["splats"]["opacities"]  # [B, N]    — 高斯不透明度
predictions["splats"]["sh"]         # [B, N, 1, 3] — 球谐函数（0 阶）
predictions["splats"]["weights"]    # [B, N]    — 逐高斯置信度权重
```

其中 `B` = 批大小（推理时始终为 1），`S` = 输入视图数，`H, W` = 图像尺寸，`N` = 总高斯数（`S × H × W`）。

---
### 先验注入
WorldMirror 2.0 接受三种几何先验作为条件输入。先验会从提供的文件中自动检测。

| 先验类型 | 条件标志 | 输入格式 |
|----------|----------|----------|
| 相机位姿 | `cond_flags[0]` | c2w 4×4 矩阵（OpenCV 约定） |
| 深度图 | `cond_flags[1]` | 逐视图浮点深度图 |
| 相机内参 | `cond_flags[2]` | 3×3 内参矩阵 |

#### 相机参数（JSON）
相机参数文件格式与 Pipeline 输出的 `camera_params.json` 一致：

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

**字段说明：**

| 字段 | 描述 |
|------|------|
| `camera_id` | 整数索引（`0`、`1`、`2` ...）或图像文件名（不含扩展名，如 `"image_0001"`） |
| `extrinsics.matrix` | 4×4 相机到世界（c2w）变换矩阵，OpenCV 坐标约定 |
| `intrinsics.matrix` | 3×3 相机内参矩阵（像素单位）：`fx, fy` 为焦距，`cx, cy` 为主点坐标 |

**重要说明：**
- `extrinsics` 和 `intrinsics` 列表可以独立提供或一起提供。列表为空 `[]` 或缺失字段表示该先验不可用。
- **内参分辨率：** 值应对应**原始图像分辨率**。Pipeline 会根据推理时的 resize + center-crop 自动调整。
- **外参对齐：** Pipeline 会自动将所有外参相对于第一帧归一化，与训练行为一致。
#### 深度图（文件夹）
深度图以独立文件存储在一个文件夹中。文件名应与输入图像文件名对应。支持格式：`.npy`、`.exr`、`.png`（16-bit）。

```
prior_depth/
├── image_0001.npy    # float32, shape [H, W]
├── image_0002.npy
└── ...
```

#### 组合先验
先验可以自由组合。示例：

```bash
# 仅内参
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_cam_path camera_intrinsics_only.json
# 仅深度
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_depth_path depth_maps/
# 相机位姿 + 内参 + 深度
python -m hyworld2.worldrecon.pipeline --input_path images/ \
    --prior_cam_path camera_params.json \
    --prior_depth_path depth_maps/
```

---
### 多卡推理
WorldMirror 2.0 支持跨多卡的**序列并行（SP）**推理，其中 token 序列在 ViT 骨干网络中跨 rank 分片，DPT 预测头并行处理帧。

> **要求：** 输入图像数量必须 **>= GPU 数量**（`nproc_per_node`）。例如，使用 8 卡时需要提供至少 8 张输入图像。如果不满足此条件，Pipeline 将报错。

```bash
# 2 卡 + FSDP + bf16
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --use_fsdp --enable_bf16
# 4 卡
torchrun --nproc_per_node=4 -m hyworld2.worldrecon.pipeline \
    --input_path path/to/images \
    --use_fsdp --enable_bf16
# Python API（在 torchrun 脚本内）
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
pipeline = WorldMirrorPipeline.from_pretrained(
    'tencent/HY-World-2.0',
    use_fsdp=True,
    enable_bf16=True,
)
pipeline('path/to/images')
```

**内部工作原理：**
1. `from_pretrained` 自动检测 `WORLD_SIZE > 1` 并初始化 `torch.distributed`。
2. 模型在 rank 0 上加载，并通过 `sync_module_states=True` 广播。
3. FSDP 将参数跨 SP 进程组分片。
4. DPT 预测头将帧分配到各 rank 并通过 `AllGather` 汇总结果。
5. 后处理（掩码计算、保存）仅在 rank 0 上运行。

---
### 高级选项
#### 禁用预测头
当只需要特定输出时，可以禁用不需要的预测头以节省显存：

```python
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

pipeline = WorldMirrorPipeline.from_pretrained(
    'tencent/HY-World-2.0',
    disable_heads=["normal", "points"],  # 释放约 200M 参数
)
```

可禁用的预测头：`"camera"`、`"depth"`、`"normal"`、`"points"`、`"gs"`。
#### 掩码过滤
Pipeline 支持三种输出过滤方式，以提高点云和高斯质量：
1. **天空掩码**（`apply_sky_mask=True`）：使用基于 ONNX 的分割模型移除天空区域，可选与模型预测的深度掩码融合。
2. **边缘掩码**（`apply_edge_mask=True`）：移除深度/法线不连续处（物体边界）的点。
3. **置信度掩码**（`apply_confidence_mask=False`）：移除预测置信度最低的 N% 的点。
这些掩码独立应用于 `points.ply`（基于深度）和 `gaussians.ply`（基于 GS）输出。GS 输出在可用时使用其自身的深度预测进行边缘检测。
#### 点云压缩
当 `compress_pts=True`（默认）时，基于深度的点云会经过以下处理：
1. **体素合并**：每个体素内的点（大小由 `compress_pts_voxel_size` 控制）通过加权平均进行合并。
2. **随机下采样**：如果结果超过 `compress_pts_max_points`，则进行均匀下采样。
类似地，高斯也会经过体素剪枝（均值、尺度、四元数、颜色、不透明度的加权平均），并可选下采样至 `compress_gs_max_points`。

---
### Gradio 应用
WorldMirror 2.0 的交互式 Web 演示。上传图像或视频，即可在浏览器中可视化 3DGS、点云、深度图、法线图和相机参数。
**快速开始：**

```bash
# 单卡
python -m hyworld2.worldrecon.gradio_app

# 多卡
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.gradio_app \
    --use_fsdp --enable_bf16
```

**使用本地检查点：**

```bash
python -m hyworld2.worldrecon.gradio_app \
    --config_path /path/to/config.yaml \
    --ckpt_path /path/to/checkpoint.safetensors
```

**创建公开链接（如 Colab 或远程服务器）：**

```bash
python -m hyworld2.worldrecon.gradio_app --share
```

**参数：**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--port` | `8081` | 服务端口 |
| `--host` | `0.0.0.0` | 服务主机 |
| `--share` | `False` | 创建公开的 Gradio 链接 |
| `--examples_dir` | `./examples/worldrecon` | 示例场景目录路径 |
| `--config_path` | `None` | 训练配置 YAML（与 `--ckpt_path` 配合使用） |
| `--ckpt_path` | `None` | 本地检查点文件（`.ckpt` / `.safetensors`） |
| `--use_fsdp` | `False` | 启用 FSDP 多卡分片 |
| `--enable_bf16` | `False` | 启用 bfloat16 混合精度 |
| `--fsdp_cpu_offload` | `False` | 将 FSDP 参数卸载到 CPU（节省显存） |

> **重要提示：** 在多卡模式下，输入图像数量必须 **>= GPU 数量**。

---
## 全景生成
*即将发布。*
本节将记录全景生成模型，包括：
- 文本到全景和图像到全景的 API
- 模型架构（基于 MMDiT 的隐式透视到 ERP 映射）
- 配置参数
- 输出格式

---
## 世界生成
*即将发布。*
本节将记录世界生成流水线，包括：
- 轨迹规划配置
- 基于记忆驱动的视频生成进行世界扩展
- 世界组合（点云扩展 + 3DGS 优化）
- 从文本/图像到可导航3D世界的端到端生成
