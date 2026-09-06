@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Leviathan ARC-AGI-1 - Strength First
set "HF_HOME=C:\LeviathanBenchmarkCache\huggingface"
set "HF_HUB_CACHE=C:\LeviathanBenchmarkCache\huggingface\hub"
set "HF_HUB_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "PY=C:\LeviathanBenchmarkCache\.venv-v7\Scripts\python.exe"
echo ARC-AGI-1 exact grids. One frozen Qwen. No training or reinstall.
echo All result files are preserved in results_strength.
echo.
"%PY%" -u "%~dp0scripts\run_strength_arc.py" --mode hybrid --split evaluation --limit 400 --symbolic-control --donor-control --activation-trials
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" echo Run stopped; the exact error is printed above and recorded in RESULTS.json.
pause
exit /b %CODE%
