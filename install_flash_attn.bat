@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
call C:\workspace\world\HY-World-2.0\.venv\Scripts\activate.bat
pip install flash-attn --no-build-isolation
