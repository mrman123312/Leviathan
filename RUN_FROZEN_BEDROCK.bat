@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Leviathan Frozen Bedrock - Existing CUDA Environment
set "HF_HOME=C:\LeviathanBenchmarkCache\huggingface"
set "HF_HUB_CACHE=C:\LeviathanBenchmarkCache\huggingface\hub"
set "HF_HUB_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "PY=C:\LeviathanBenchmarkCache\.venv-v7\Scripts\python.exe"
echo Uses your working v7 Python/CUDA and cached Qwen. No installs. No training.
echo.
"%PY%" -u "%~dp0scripts\run_bedrock_lab.py" --output "%~dp0results"
set "RESULT=%ERRORLEVEL%"
if exist "%~dp0results\RESULTS.html" start "" "%~dp0results\RESULTS.html"
echo.
if not "%RESULT%"=="0" echo The actual error is printed above and saved in results\RESULTS.json.
pause
exit /b %RESULT%
