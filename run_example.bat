@echo off
:: Run HY-World 2.0 bundled examples for each open-sourced module.
::
:: Modules covered:
::   --module worldrecon  (default)  WorldMirror 2.0 reconstruction (Park example).
::                                    Always runs out-of-the-box.
::   --module panogen                HY-Pano-2.0 panorama generation. REQUIRES:
::                                     1. HY-Pano-2.0 weights downloaded (set HY_PANO_MODEL_DIR)
::                                     2. an input image (set HY_PANO_INPUT_PNG)
::   --module worldgen               WorldStereo-2 + WorldNav 5-stage pipeline.
::                                   NOT covered here — it needs an external vLLM
::                                   server + custom CUDA gsplat/navmesh builds.
::                                   See hyworld2/worldgen/README.md.
::   --module all                    Runs worldrecon, then panogen if both env
::                                   vars set (else warns and skips it).
::
:: Examples:
::   run_example.bat                              worldrecon (default)
::   run_example.bat --module worldrecon
::   set HY_PANO_MODEL_DIR=C:\models\HY-Pano-2.0
::   set HY_PANO_INPUT_PNG=C:\some\image.png
::   run_example.bat --module panogen
::   run_example.bat --module all

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat

set MODULE=all
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--module" ( set MODULE=%~2 & shift & shift & goto parse_args )
shift
goto parse_args
:args_done

if /I "%MODULE%"=="all"        goto run_all
if /I "%MODULE%"=="worldrecon" goto run_worldrecon
if /I "%MODULE%"=="panogen"    goto run_panogen
if /I "%MODULE%"=="worldgen"   goto run_worldgen
echo ERROR: unknown module %MODULE%. Use worldrecon ^| panogen ^| worldgen ^| all.
exit /b 2

:run_worldrecon
echo === WorldMirror 2.0 reconstruction (examples\worldrecon\realistic\Park) ===
python -m hyworld2.worldrecon.pipeline ^
    --input_path examples\worldrecon\realistic\Park ^
    --output_path output\park ^
    --save_rendered ^
    --render_interp_per_pair 15 ^
    --enable_bf16
if errorlevel 1 ( echo FAIL worldrecon rc=%ERRORLEVEL% & exit /b %ERRORLEVEL% )
if /I not "%MODULE%"=="all" exit /b 0

:run_panogen
echo.
echo === HY-Pano 2.0 panorama generation ===
if not defined HY_PANO_MODEL_DIR (
    echo SKIP panogen: HY_PANO_MODEL_DIR not set ^(point at HY-Pano-2.0 weights^).
    if /I "%MODULE%"=="all" goto :eof
    exit /b 2
)
if not defined HY_PANO_INPUT_PNG (
    echo SKIP panogen: HY_PANO_INPUT_PNG not set ^(point at an input image^).
    if /I "%MODULE%"=="all" goto :eof
    exit /b 2
)
if not defined HY_PANO_PROMPT set HY_PANO_PROMPT=Expand this image to a 360-degree equirectangular panorama. Maintain realistic style.
pushd hyworld2\panogen
python pipeline.py ^
    --pretrained-model-name-or-path "%HY_PANO_MODEL_DIR%" ^
    --subfolder "" ^
    --image "%HY_PANO_INPUT_PNG%" ^
    --prompt "%HY_PANO_PROMPT%" ^
    --save "%~dp0output\panorama.png"
set RC=%ERRORLEVEL%
popd
if not %RC%==0 ( echo FAIL panogen rc=%RC% & exit /b %RC% )
exit /b 0

:run_worldgen
echo.
echo SKIP worldgen: multi-stage pipeline ^(traj_generate -^> traj_render -^> video_gen
echo                 -^> gen_gs_data -^> world_gs_trainer^). Needs an external vLLM
echo                 server ^(Qwen3-VL-8B^) and custom CUDA builds ^(gsplat / navmesh^).
echo                 See hyworld2\worldgen\README.md for the full workflow.
exit /b 0

:run_all
call :run_worldrecon
call :run_panogen
exit /b 0

endlocal
