@echo off
call .venv\Scripts\activate.bat

python run_vbench.py C:/Users/kschmid/.cache/huggingface/hub/models--tencent--HY-World-2.0/snapshots/776d5a92c860c25105c62e5d264beef8ac39bcbb ^
    --output_dir results_vbench/videos ^
    --num_samples 1 ^
    --image_types "indoor,scenery" ^
    --target_size 952 ^
    --render_interp_per_pair 15 ^
    --enable_bf16 True
