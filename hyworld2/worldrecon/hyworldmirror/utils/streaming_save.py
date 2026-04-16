"""Bounded-memory artifact writers for WorldMirror inference outputs."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from ..models.utils.camera_utils import vector_to_camera_matrices
from ..models.utils.geometry import depth_to_world_coords_points
from .output_budget import OutputSavePlan, build_output_save_plan


_GS_ATTRIBUTES = (
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)
_GS_ROW_DTYPE = np.dtype([(name, "<f4") for name in _GS_ATTRIBUTES])
_POINT_ROW_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def save_gaussian_splats_artifact(
    path: Path,
    splats: Dict[str, torch.Tensor],
    *,
    filter_mask: Optional[np.ndarray] = None,
    gs_filter_mask: Optional[np.ndarray] = None,
    max_points: int = 5_000_000,
    voxel_size: float = 0.002,
    memory_budget_gb: Optional[float] = 4.0,
    save_chunk_frames: Optional[int] = None,
    image_shape: Optional[Tuple[int, int, int]] = None,
    quantile_threshold: float = 0.98,
) -> Dict[str, float]:
    """Save 3D Gaussian splats without materializing all source splats on CPU."""
    path = Path(path)
    t0 = time.perf_counter()

    means_t = splats["means"][0].reshape(-1, 3)
    scales_t = splats["scales"][0].reshape(-1, 3)
    quats_t = splats["quats"][0].reshape(-1, 4)
    colors_t = (splats["sh"][0] if "sh" in splats else splats["colors"][0]).reshape(-1, 3)
    opacities_t = splats["opacities"][0].reshape(-1)
    weights_t = (
        splats["weights"][0].reshape(-1)
        if "weights" in splats
        else torch.ones_like(opacities_t)
    )

    total = int(means_t.shape[0])
    plan = _resolve_plan(
        total,
        image_shape=image_shape,
        memory_budget_gb=memory_budget_gb,
        save_chunk_frames=save_chunk_frames,
    )

    keep = _flatten_keep_mask(gs_filter_mask if gs_filter_mask is not None else filter_mask)
    if keep is not None and keep.shape[0] != total:
        raise ValueError(f"Gaussian filter mask has {keep.shape[0]} entries, expected {total}")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive for Gaussian voxel pruning")
    aggregated = []
    for start, end in _iter_ranges(total, plan.gaussian_chunk_elements):
        sl = slice(start, end)
        keep_chunk = keep[sl] if keep is not None else None
        arrays = _copy_gaussian_chunk(
            means_t[sl], scales_t[sl], quats_t[sl], colors_t[sl],
            opacities_t[sl], weights_t[sl], keep_chunk,
        )
        if arrays is None:
            continue
        aggregated.append(_aggregate_gaussians_by_voxel(*arrays, voxel_size=voxel_size))

    if aggregated:
        keys, means, scales, quats, colors, opacities, *_ = _finalize_gaussian_aggregates(aggregated)
        del keys
    else:
        means = np.empty((0, 3), dtype=np.float32)
        scales = np.empty((0, 3), dtype=np.float32)
        quats = np.empty((0, 4), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.float32)
        opacities = np.empty((0,), dtype=np.float32)

    if max_points > 0 and means.shape[0] > max_points:
        idx = np.random.default_rng(42).choice(means.shape[0], size=max_points, replace=False)
        means, scales, quats = means[idx], scales[idx], quats[idx]
        colors, opacities = colors[idx], opacities[idx]

    written = _write_gaussian_ply_binary(
        path, means, scales, quats, colors, opacities,
        quantile_threshold=quantile_threshold,
        chunk_rows=plan.gaussian_chunk_elements,
    )
    return {
        "save_gs_ply": time.perf_counter() - t0,
        "gs_source_count": float(total),
        "gs_written_count": float(written),
        "gs_chunks": float(plan.gaussian_chunks),
        "gs_estimated_source_gb": plan.estimated_gaussian_source_gb,
    }


def save_points_artifact(
    path: Path,
    predictions: Dict[str, torch.Tensor],
    imgs: torch.Tensor,
    *,
    filter_mask: Optional[np.ndarray] = None,
    compress: bool = True,
    max_points: int = 2_000_000,
    voxel_size: float = 0.002,
    memory_budget_gb: Optional[float] = 4.0,
    save_chunk_frames: Optional[int] = None,
) -> Dict[str, float]:
    """Save depth-derived point cloud with bounded host memory."""
    path = Path(path)
    t0 = time.perf_counter()
    _, S, _, H, W = imgs.shape
    if filter_mask is not None and np.asarray(filter_mask).shape != (S, H, W):
        raise ValueError(f"filter_mask must have shape {(S, H, W)}, got {np.asarray(filter_mask).shape}")
    if compress and voxel_size <= 0:
        raise ValueError("voxel_size must be positive when compress=True")
    plan = build_output_save_plan(
        S, H, W,
        memory_budget_gb=memory_budget_gb,
        save_chunk_frames=save_chunk_frames,
    )

    e3x4, intr = vector_to_camera_matrices(predictions["camera_params"], image_hw=(H, W))
    depth = predictions["depth"].float()
    imgs_f = imgs.float()
    extrinsics = e3x4[0].float()
    intrinsics = intr[0].float()

    if compress:
        aggregates = []
        for start in range(0, S, plan.point_frame_chunk_size):
            end = min(S, start + plan.point_frame_chunk_size)
            pts_np, cols_np = _compute_points_chunk(
                depth, imgs_f, extrinsics, intrinsics, start, end, filter_mask=filter_mask
            )
            if pts_np.shape[0] == 0:
                continue
            aggregates.append(_aggregate_points_by_voxel(pts_np, cols_np, voxel_size=voxel_size))

        if aggregates:
            keys, sums, color_sums, counts = _merge_point_aggregates(aggregates)
            _, pts_np, cols_np = _finalize_point_aggregate(keys, sums, color_sums, counts)
        else:
            pts_np = np.empty((0, 3), dtype=np.float32)
            cols_np = np.empty((0, 3), dtype=np.uint8)

        if max_points > 0 and pts_np.shape[0] > max_points:
            idx = np.random.default_rng(42).choice(pts_np.shape[0], size=max_points, replace=False)
            pts_np, cols_np = pts_np[idx], cols_np[idx]
        t_compress_done = time.perf_counter()
        written = _write_points_ply_binary(path, pts_np, cols_np)
        compress_time = t_compress_done - t0
    else:
        written = _write_points_uncompressed_stream(
            path, depth, imgs_f, extrinsics, intrinsics, S, filter_mask=filter_mask
        )
        compress_time = 0.0

    return {
        "save_points": time.perf_counter() - t0,
        "compress_points": compress_time,
        "points_written_count": float(written),
        "point_frame_chunk_size": float(plan.point_frame_chunk_size),
    }


def _resolve_plan(
    total_elements: int,
    *,
    image_shape: Optional[Tuple[int, int, int]],
    memory_budget_gb: Optional[float],
    save_chunk_frames: Optional[int],
) -> OutputSavePlan:
    if image_shape is not None:
        S, H, W = image_shape
    else:
        S, H, W = 1, 1, max(1, total_elements)
    return build_output_save_plan(
        S, H, W,
        memory_budget_gb=memory_budget_gb,
        save_chunk_frames=save_chunk_frames,
    )


def _flatten_keep_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    return np.asarray(mask, dtype=bool).reshape(-1)


def _iter_ranges(total: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, total, max(1, int(chunk_size))):
        yield start, min(total, start + int(chunk_size))


def _copy_gaussian_chunk(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    weights: torch.Tensor,
    keep: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    means_np = means.detach().cpu().float().numpy()
    scales_np = scales.detach().cpu().float().numpy()
    quats_np = quats.detach().cpu().float().numpy()
    colors_np = colors.detach().cpu().float().numpy()
    opacities_np = opacities.detach().cpu().float().numpy()
    weights_np = weights.detach().cpu().float().numpy()

    if keep is not None:
        if not keep.any():
            return None
        means_np = means_np[keep]
        scales_np = scales_np[keep]
        quats_np = quats_np[keep]
        colors_np = colors_np[keep]
        opacities_np = opacities_np[keep]
        weights_np = weights_np[keep]

    finite = (
        np.isfinite(means_np).all(axis=1)
        & np.isfinite(scales_np).all(axis=1)
        & np.isfinite(quats_np).all(axis=1)
        & np.isfinite(colors_np).all(axis=1)
        & np.isfinite(opacities_np)
        & np.isfinite(weights_np)
    )
    if not finite.all():
        means_np = means_np[finite]
        scales_np = scales_np[finite]
        quats_np = quats_np[finite]
        colors_np = colors_np[finite]
        opacities_np = opacities_np[finite]
        weights_np = weights_np[finite]
    if means_np.shape[0] == 0:
        return None
    return means_np, scales_np, quats_np, colors_np, opacities_np, weights_np


def _aggregate_gaussians_by_voxel(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    weights: np.ndarray,
    *,
    voxel_size: float,
) -> Tuple[np.ndarray, ...]:
    keys = np.floor(means / voxel_size).astype(np.int64)
    return _aggregate_gaussians_by_keys(keys, means, scales, quats, colors, opacities, weights)


def _aggregate_gaussians_by_keys(
    keys: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    if weights is None:
        weights = np.ones(opacities.shape[0], dtype=np.float32)
    weights = weights.astype(np.float32, copy=False)

    unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    count = unique_keys.shape[0]
    group_counts = np.bincount(inv, minlength=count).astype(np.int64)
    wsum = np.bincount(inv, weights=weights, minlength=count).astype(np.float32)
    wsum = np.maximum(wsum, 1e-8)

    means_sum = _weighted_sum(inv, means, weights, count)
    scales_sum = _weighted_sum(inv, scales, weights, count)
    quats_sum = _weighted_sum(inv, quats, weights, count)
    colors_sum = _weighted_sum(inv, colors, weights, count)
    opacity_weight_sums = np.bincount(inv, weights=weights * weights, minlength=count).astype(np.float32)
    opacity_original_sums = np.bincount(inv, weights=opacities, minlength=count).astype(np.float32)

    means_out = means_sum / wsum[:, None]
    scales_out = scales_sum / wsum[:, None]
    quats_out = quats_sum / wsum[:, None]
    quat_norms = np.linalg.norm(quats_out, axis=1, keepdims=True)
    quats_out = quats_out / np.maximum(quat_norms, 1e-8)
    colors_out = colors_sum / wsum[:, None]
    if int(group_counts.sum()) == count:
        opacities_out = opacity_original_sums.astype(np.float32)
    else:
        opacities_out = (opacity_weight_sums / wsum).astype(np.float32)
    return (
        unique_keys, means_out, scales_out, quats_out, colors_out,
        opacities_out, wsum, opacity_weight_sums, group_counts,
        opacity_original_sums,
    )


def _weighted_sum(inv: np.ndarray, values: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1).astype(np.float32, copy=False)
    out = np.zeros((count, flat.shape[1]), dtype=np.float32)
    np.add.at(out, inv, flat * weights[:, None])
    return out.reshape((count, *values.shape[1:]))


def _merge_gaussian_aggregates(
    aggregates,
) -> Tuple[np.ndarray, ...]:
    return tuple(np.concatenate(parts, axis=0) for parts in zip(*aggregates))


def _finalize_gaussian_aggregates(aggregates) -> Tuple[np.ndarray, ...]:
    keys, means, scales, quats, colors, _, weights, opacity_weight_sums, group_counts, opacity_original_sums = (
        _merge_gaussian_aggregates(aggregates)
    )
    unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    count = unique_keys.shape[0]
    wsum = np.bincount(inv, weights=weights, minlength=count).astype(np.float32)
    wsum = np.maximum(wsum, 1e-8)
    total_counts = np.bincount(inv, weights=group_counts, minlength=count).astype(np.int64)
    total_opacity_weight_sums = np.bincount(inv, weights=opacity_weight_sums, minlength=count).astype(np.float32)
    total_opacity_original_sums = np.bincount(inv, weights=opacity_original_sums, minlength=count).astype(np.float32)

    means_out = _weighted_sum(inv, means, weights, count) / wsum[:, None]
    scales_out = _weighted_sum(inv, scales, weights, count) / wsum[:, None]
    quats_out = _weighted_sum(inv, quats, weights, count) / wsum[:, None]
    quats_out = quats_out / np.maximum(np.linalg.norm(quats_out, axis=1, keepdims=True), 1e-8)
    colors_out = _weighted_sum(inv, colors, weights, count) / wsum[:, None]
    if int(total_counts.sum()) == count:
        opacities_out = total_opacity_original_sums.astype(np.float32)
    else:
        opacities_out = (total_opacity_weight_sums / wsum).astype(np.float32)
    return (
        unique_keys, means_out, scales_out, quats_out, colors_out,
        opacities_out, wsum, total_opacity_weight_sums, total_counts,
        total_opacity_original_sums,
    )


def _write_gaussian_ply_binary(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    *,
    quantile_threshold: float = 0.98,
    chunk_rows: int = 1_000_000,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if means.shape[0] == 0:
        keep = np.zeros((0,), dtype=bool)
    else:
        scale_max = scales.max(axis=1)
        threshold = np.quantile(scale_max, quantile_threshold)
        keep = scale_max <= threshold
    written = int(keep.sum())

    with open(path, "wb") as f:
        _write_gaussian_header(f, written)
        keep_idx = np.flatnonzero(keep)
        for start, end in _iter_ranges(len(keep_idx), chunk_rows):
            idx = keep_idx[start:end]
            rows = np.empty(len(idx), dtype=_GS_ROW_DTYPE)
            rows["x"], rows["y"], rows["z"] = means[idx, 0], means[idx, 1], means[idx, 2]
            rows["nx"] = 0.0
            rows["ny"] = 0.0
            rows["nz"] = 0.0
            rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = colors[idx, 0], colors[idx, 1], colors[idx, 2]
            rows["opacity"] = opacities[idx]
            safe_scales = np.maximum(scales[idx], np.finfo(np.float32).tiny)
            log_scales = np.log(safe_scales)
            rows["scale_0"], rows["scale_1"], rows["scale_2"] = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]
            rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = (
                quats[idx, 0], quats[idx, 1], quats[idx, 2], quats[idx, 3]
            )
            f.write(rows.tobytes())
    return written


def _write_gaussian_header(handle, count: int) -> None:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
        *[f"property float {name}" for name in _GS_ATTRIBUTES],
        "end_header",
        "",
    ]
    handle.write("\n".join(lines).encode("ascii"))


def _compute_points_chunk(
    depth: torch.Tensor,
    imgs: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_start: int,
    frame_end: int,
    *,
    filter_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    points_list = []
    colors_list = []
    for i in range(frame_start, frame_end):
        d = depth[0, i, :, :, 0]
        homo = torch.tensor(
            [[0, 0, 0, 1]],
            dtype=extrinsics.dtype,
            device=extrinsics.device,
        )
        w2c = torch.cat([extrinsics[i][:3, :4], homo], dim=0)
        c2w = torch.linalg.inv(w2c)[:3, :4]
        pts_i, _, mask = depth_to_world_coords_points(d[None], c2w[None], intrinsics[i][None])
        img_colors = (imgs[0, i].permute(1, 2, 0) * 255).to(torch.uint8)
        valid = mask[0]
        if filter_mask is not None:
            valid = valid & torch.from_numpy(filter_mask[i]).to(valid.device)
        if valid.any():
            points_list.append(pts_i[0][valid].detach().cpu().float().numpy())
            colors_list.append(img_colors[valid].detach().cpu().numpy())

    if not points_list:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return np.concatenate(points_list, axis=0), np.concatenate(colors_list, axis=0)


def _aggregate_points_by_voxel(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.floor(points / voxel_size).astype(np.int64)
    unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    count = unique_keys.shape[0]
    counts = np.bincount(inv, minlength=count).astype(np.float32)
    sums = np.zeros((count, 3), dtype=np.float32)
    color_sums = np.zeros((count, 3), dtype=np.float32)
    np.add.at(sums, inv, points.astype(np.float32, copy=False))
    np.add.at(color_sums, inv, colors.astype(np.float32, copy=False))
    return unique_keys, sums, color_sums, counts


def _merge_point_aggregates(
    aggregates,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.concatenate([a[0] for a in aggregates], axis=0)
    sums = np.concatenate([a[1] for a in aggregates], axis=0)
    color_sums = np.concatenate([a[2] for a in aggregates], axis=0)
    counts = np.concatenate([a[3] for a in aggregates], axis=0)
    unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    count = unique_keys.shape[0]
    merged_sums = np.zeros((count, 3), dtype=np.float32)
    merged_color_sums = np.zeros((count, 3), dtype=np.float32)
    merged_counts = np.zeros((count,), dtype=np.float32)
    np.add.at(merged_sums, inv, sums)
    np.add.at(merged_color_sums, inv, color_sums)
    np.add.at(merged_counts, inv, counts)
    return unique_keys, merged_sums, merged_color_sums, merged_counts


def _finalize_point_aggregate(
    keys: np.ndarray,
    sums: np.ndarray,
    color_sums: np.ndarray,
    counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_counts = np.maximum(counts, 1.0)
    points = sums / safe_counts[:, None]
    colors = np.clip(np.round(color_sums / safe_counts[:, None]), 0, 255).astype(np.uint8)
    return keys, points.astype(np.float32, copy=False), colors


def _write_points_ply_binary(path: Path, points: np.ndarray, colors: np.ndarray) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(points.shape[0])
    with open(path, "wb") as f:
        _write_points_header(f, count)
        for start, end in _iter_ranges(count, 1_000_000):
            rows = np.empty(end - start, dtype=_POINT_ROW_DTYPE)
            pts = points[start:end]
            cols = colors[start:end]
            rows["x"], rows["y"], rows["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            rows["red"], rows["green"], rows["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
            f.write(rows.tobytes())
    return count


def _write_points_uncompressed_stream(
    path: Path,
    depth: torch.Tensor,
    imgs: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    num_frames: int,
    *,
    filter_mask: Optional[np.ndarray],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
            tmp_path = tmp.name
            for i in range(num_frames):
                pts_np, cols_np = _compute_points_chunk(
                    depth, imgs, extrinsics, intrinsics, i, i + 1, filter_mask=filter_mask
                )
                total += _write_point_rows(tmp, pts_np, cols_np)
        with open(path, "wb") as out, open(tmp_path, "rb") as body:
            _write_points_header(out, total)
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return total


def _write_point_rows(handle, points: np.ndarray, colors: np.ndarray) -> int:
    rows = np.empty(points.shape[0], dtype=_POINT_ROW_DTYPE)
    rows["x"], rows["y"], rows["z"] = points[:, 0], points[:, 1], points[:, 2]
    rows["red"], rows["green"], rows["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    handle.write(rows.tobytes())
    return int(points.shape[0])


def _write_points_header(handle, count: int) -> None:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ]
    handle.write("\n".join(lines).encode("ascii"))
