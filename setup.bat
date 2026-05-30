@echo off
echo === HY-World-2.0 Setup ===

:: Create venv
if not exist .venv (
    echo Creating .venv...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

:: PyTorch 2.7 + CUDA 12.8 (supports RTX 5090 sm_120 / Blackwell)
echo Installing PyTorch 2.7.0 cu128...
.venv\Scripts\pip.exe install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --quiet

:: All other requirements (gsplat line is commented out — built below)
echo Installing requirements...
pip install -r requirements.txt --quiet

:: Build flash-attn and gsplat from source (require --no-build-isolation + MSVC)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

echo Installing flash-attn (pre-built wheel for cp312+cu128+torch2.7.0)...
.venv\Scripts\pip.exe install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%%2Bcu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl"

echo Building gsplat v1.5.3 from source (no pre-built Windows/cp312 wheel)...
.venv\Scripts\pip.exe install C:\workspace\world\gsplat --no-build-isolation < NUL

echo.
echo === Setup complete ===
