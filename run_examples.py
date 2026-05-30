"""Run WorldMirror 2.0 on all examples/worldrecon examples and save rendered videos."""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

EXAMPLES_ROOT = Path(__file__).parent / "examples" / "worldrecon"
OUTPUT_ROOT   = Path(__file__).parent / "output" / "examples"
MODEL_PATH    = snapshot_download(
    repo_id="tencent/HY-World-2.0",
    allow_patterns=["HY-WorldMirror-2.0/*"],
    local_files_only=True,
)

# Leaf directories = contain images but no sub-directories
leaf_dirs = sorted(
    p for p in EXAMPLES_ROOT.rglob("*")
    if p.is_dir() and not any(c.is_dir() for c in p.iterdir())
)

print(f"Found {len(leaf_dirs)} examples")
for d in leaf_dirs:
    print(f"  {d.relative_to(EXAMPLES_ROOT)}")

pipeline = WorldMirrorPipeline.from_pretrained(MODEL_PATH, enable_bf16=True)

for example_dir in leaf_dirs:
    rel = example_dir.relative_to(EXAMPLES_ROOT)
    out = OUTPUT_ROOT / rel
    print(f"\n--- {rel} ---")
    result_dir = pipeline(
        str(example_dir),
        output_path=str(out),
        save_rendered=True,
        render_interp_per_pair=15,
        save_depth=True,
        save_normal=True,
        save_gs=True,
        save_points=True,
    )
    video = Path(result_dir) / "rendered" / "rendered_rgb.mp4"
    if not video.is_file():
        raise RuntimeError(f"Rendered video not generated for {rel} — expected: {video}")
    print(f"[OK] video: {video}")

print("\nAll examples done. Output in:", OUTPUT_ROOT)
