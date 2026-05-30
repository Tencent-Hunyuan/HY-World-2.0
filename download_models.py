"""
Download every model HY-World-2.0 can need into the local HF cache, using a
per-file serial loop. This bypasses ``huggingface_hub.snapshot_download``,
whose tqdm.contrib.concurrent ThreadPoolExecutor shutdown races on Windows
("cannot join thread before it is started") even at max_workers=1.

Modules and their model requirements:
  - worldrecon   : tencent/HY-World-2.0          subfolder HY-WorldMirror-2.0
  - panogen      : tencent/HY-World-2.0          subfolder HY-Pano-2.0
  - worldgen     : ewrfcas/Uni3C                 (full repo)
                   Ruicheng/moge-2-vitl-normal   (full repo)
                   facebook/sam3                 (full repo)

Usage:
    python download_models.py                 # all models needed for all modules
    python download_models.py --module worldrecon
    python download_models.py --module panogen
    python download_models.py --module worldgen
    python download_models.py --model worldmirror
    python download_models.py --local_dir ./checkpoints
"""

import argparse
import fnmatch
import os
from huggingface_hub import HfApi, hf_hub_download

# (key, repo_id, subfolder_or_None, description)
MODELS: dict[str, dict] = {
    "worldmirror": {
        "repo_id": "tencent/HY-World-2.0",
        "subfolder": "HY-WorldMirror-2.0",
        "module": "worldrecon",
        "description": "WorldMirror 2.0 — multi-view/video -> 3D reconstruction (~1.2B params)",
    },
    "panogen": {
        "repo_id": "tencent/HY-World-2.0",
        "subfolder": "HY-Pano-2.0",
        "module": "panogen",
        "description": "HY-Pano 2.0 — panorama generation (HunyuanImage-3 fine-tune)",
    },
    "uni3c": {
        "repo_id": "ewrfcas/Uni3C",
        "subfolder": None,
        "module": "worldgen",
        "description": "Uni3C ControlNet — used by WorldStereo worldgen",
    },
    "moge": {
        "repo_id": "Ruicheng/moge-2-vitl-normal",
        "subfolder": None,
        "module": "worldgen",
        "description": "MoGe-2 ViT-L/16 normal — depth/normal estimator for gs data",
    },
    "sam3": {
        "repo_id": "facebook/sam3",
        "subfolder": None,
        "module": "worldgen",
        "description": "SAM3 — segmentation used in worldgen trajectory generation",
    },
}

MODULE_TO_KEYS: dict[str, list[str]] = {}
for k, v in MODELS.items():
    MODULE_TO_KEYS.setdefault(v["module"], []).append(k)


def download_one(key: str, local_dir: str | None = None) -> str:
    info = MODELS[key]
    repo_id, subfolder = info["repo_id"], info["subfolder"]
    print(f"\n--- {key}: {info['description']}")
    print(f"    repo={repo_id}  subfolder={subfolder or '(whole repo)'}")

    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id)
    if subfolder:
        pattern = f"{subfolder}/*"
        files = [f for f in all_files if fnmatch.fnmatch(f, pattern)]
    else:
        files = list(all_files)
    if not files:
        raise RuntimeError(f"no files matched for {key} ({repo_id} subfolder={subfolder})")

    print(f"    {len(files)} file(s) to fetch")
    last_path = None
    for i, rel in enumerate(files, 1):
        print(f"    [{i:>3}/{len(files)}] {rel}")
        last_path = hf_hub_download(repo_id=repo_id, filename=rel, local_dir=local_dir)

    if local_dir:
        resolved = os.path.join(local_dir, subfolder) if subfolder else local_dir
    else:
        # last_path is …/snapshots/<rev>/<rel>; strip <rel> to get the snapshot root,
        # then append subfolder.
        snap_root = last_path
        for _ in (rel.split("/") if False else range(rel.count("/") + 1)):
            snap_root = os.path.dirname(snap_root)
        resolved = os.path.join(snap_root, subfolder) if subfolder else snap_root
    print(f"    saved -> {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HY-World-2.0 model weights")
    parser.add_argument(
        "--module",
        choices=sorted(MODULE_TO_KEYS) + ["all"],
        default="all",
        help="Which module's models to fetch (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS) + ["all"],
        default=None,
        help="Override --module: pick exactly one model key",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Optional local directory (default: HuggingFace cache)",
    )
    args = parser.parse_args()

    if args.model:
        keys = [args.model] if args.model != "all" else list(MODELS)
    elif args.module == "all":
        keys = list(MODELS)
    else:
        keys = MODULE_TO_KEYS[args.module]

    print(f"Will fetch {len(keys)} model(s): {keys}")
    for key in keys:
        download_one(key, args.local_dir)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
