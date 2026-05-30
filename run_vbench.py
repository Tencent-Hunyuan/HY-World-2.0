"""
VBench batch runner for HY-World-2.0 (WorldMirror 2.0).

WorldMirror 2.0 is a 3D reconstruction model (multi-view → 3DGS).
For each VBench image, we run reconstruction and render a fly-through
video from the resulting Gaussian splats, which is saved as the VBench
output video.

Usage:
    python run_vbench.py

    # Override model path:
    python run_vbench.py tencent/HY-World-2.0 ^
        --output_dir results_vbench/videos
"""

import csv
import json
import os
import re
import shutil
import tempfile
import time

import fire
import psutil
import torch

_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
_VBENCH_ROOT       = os.path.join(_SCRIPT_DIR, "..", "VBench", "vbench2_beta_i2v", "vbench2_beta_i2v", "data")
_DEFAULT_INFO_JSON = os.path.join(_VBENCH_ROOT, "i2v-bench-info.json")
_DEFAULT_CROP_DIR  = os.path.join(_VBENCH_ROOT, "crop")


def _safe(prompt):
    return re.sub(r'[<>:"/\\|?*]', "_", prompt)[:150]


_DEFAULT_MODEL = "C:/Users/kschmid/.cache/huggingface/hub/models--tencent--HY-World-2.0/snapshots/776d5a92c860c25105c62e5d264beef8ac39bcbb"


def vbench_batch(
    model_path=_DEFAULT_MODEL,
    output_dir="results_vbench/videos",
    num_samples=1,
    image_types="indoor,scenery",
    resolution="1-1",
    target_size=952,
    render_interp_per_pair=15,
    enable_bf16=True,
    vbench_info_json=None,
    crop_dir=None,
):
    """Run WorldMirror 2.0 reconstruction + fly-through render on VBench images.

    Args:
        model_path: HuggingFace repo ID or local path to model checkpoint.
        output_dir: Directory to save rendered videos.
        num_samples: Samples per prompt (WorldMirror is deterministic; 1 recommended).
        image_types: Comma-separated VBench image types to include (e.g. "indoor,scenery").
        resolution: VBench crop resolution key (e.g. "1-1", "16-9").
        target_size: Max image resolution for WorldMirror inference.
        render_interp_per_pair: Interpolated frames per camera pair in fly-through video.
        enable_bf16: Use bf16 precision.
        vbench_info_json: Override path to i2v-bench-info.json.
        crop_dir: Override path to VBench crop image directory.
    """
    info_json = os.path.abspath(vbench_info_json or _DEFAULT_INFO_JSON)
    crop_base = os.path.abspath(crop_dir or _DEFAULT_CROP_DIR)
    image_dir = os.path.join(crop_base, resolution)
    out_dir   = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path    = os.path.join(os.path.dirname(out_dir), "vbench_stats.csv")
    stats_is_new  = not os.path.exists(stats_path)
    stats_f       = open(stats_path, "a", newline="", encoding="utf-8")
    stats_w       = csv.writer(stats_f)
    if stats_is_new:
        stats_w.writerow(["task_idx", "prompt", "sample_idx", "duration_s", "ram_gb", "vram_gb", "out_path", "status"])

    if not os.path.isfile(info_json):
        print(f"[vbench] ERROR: info JSON not found: {info_json}"); return
    if not os.path.isdir(image_dir):
        print(f"[vbench] ERROR: crop dir not found: {image_dir}"); return

    with open(info_json, encoding="utf-8") as f:
        entries = json.load(f)

    if isinstance(image_types, (list, tuple)):
        allowed = {t.strip() for t in image_types if t.strip()} or None
    else:
        allowed = {t.strip() for t in image_types.split(",") if t.strip()} if image_types else None
    seen, prompts = set(), []
    for e in entries:
        name = e["file_name"]
        if name in seen:
            continue
        if allowed and e.get("type") not in allowed:
            continue
        seen.add(name)
        caption = e.get("caption", os.path.splitext(name)[0])
        prompts.append((name, caption))

    total = len(prompts) * num_samples
    print(f"[vbench] {len(prompts)} prompts × {num_samples} samples = {total} total")

    # Load pipeline once
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
    pipeline = WorldMirrorPipeline.from_pretrained(model_path, enable_bf16=enable_bf16)

    skipped = generated = errors = 0
    done = 0
    t_start = time.time()

    for task_idx, (image_name, prompt) in enumerate(prompts):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"[vbench] skip {task_idx}: image not found — {image_path}")
            continue

        for sample_idx in range(num_samples):
            out_path = os.path.join(out_dir, f"{_safe(prompt)}-{sample_idx}.mp4")
            if os.path.exists(out_path):
                skipped += 1
                done += 1
                stats_w.writerow([task_idx, prompt, sample_idx, "", "", "", out_path, "skipped"])
                stats_f.flush()
                continue

            pct = 100 * done / total if total else 0
            eta = ""
            if done > 0:
                elapsed = time.time() - t_start
                secs_left = elapsed / done * (total - done)
                eta = f"  ETA {int(secs_left//3600):02d}h{int(secs_left%3600//60):02d}m{int(secs_left%60):02d}s"
            print(f"[vbench] [{done+1}/{total}  {pct:.0f}%{eta}]  prompt {task_idx+1}/{len(prompts)}  sample {sample_idx+1}/{num_samples}: {prompt[:60]}")

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    dst = os.path.join(tmpdir, image_name)
                    shutil.copy2(image_path, dst)

                    with tempfile.TemporaryDirectory() as workdir:
                        st = time.time()
                        pipeline(
                            tmpdir,
                            output_path=workdir,
                            target_size=target_size,
                            save_rendered=True,
                            render_interp_per_pair=render_interp_per_pair,
                            save_depth=False,
                            save_normal=False,
                            save_gs=True,
                            save_points=False,
                            save_camera=False,
                            log_time=False,
                            strict_output_path=os.path.join(workdir, "out"),
                        )
                        ed = time.time()

                        rendered = os.path.join(workdir, "out", "rendered", "rendered_rgb.mp4")
                        if os.path.isfile(rendered):
                            shutil.copy2(rendered, out_path)
                            status = "ok"
                        else:
                            raise RuntimeError(f"Rendered video not produced for {image_name} — gsplat_cuda may not be built correctly.")

                ram_gb  = psutil.Process().memory_info().rss / 1024**3
                vram_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                dur = ed - st
                print(f"[vbench] saved  {out_path}  ({dur:.1f}s  RAM {ram_gb:.1f}GB  VRAM {vram_gb:.1f}GB)")
                stats_w.writerow([task_idx, prompt, sample_idx, f"{dur:.2f}", f"{ram_gb:.2f}", f"{vram_gb:.2f}", out_path, status])
                stats_f.flush()
                if status == "ok":
                    generated += 1
                else:
                    errors += 1
            except Exception as exc:
                print(f"[vbench] ERROR task {task_idx} sample {sample_idx}: {exc}")
                stats_w.writerow([task_idx, prompt, sample_idx, "", "", "", out_path, "error"])
                stats_f.flush()
                errors += 1
            done += 1

    elapsed_total = time.time() - t_start
    stats_f.close()
    print(f"\n[vbench] done — generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed_total/60:.1f}m")
    print(f"[vbench] stats → {stats_path}")


if __name__ == "__main__":
    fire.Fire(vbench_batch)
