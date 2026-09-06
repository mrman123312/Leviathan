@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Leviathan v3 - ARC-Easy and Frozen Architecture
set "HF_HOME=C:\LeviathanBenchmarkCache\huggingface"
set "HF_HUB_CACHE=C:\LeviathanBenchmarkCache\huggingface\hub"
set "HF_HUB_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "PYTHONUTF8=1"
set "PY=C:\LeviathanBenchmarkCache\.venv-v7\Scripts\python.exe"
"%PY%" -u "%~dp0scripts\launch_bedrock_v3.py"
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo The run did not fully complete. The exact error and partial results are retained.
pause
exit /b %RESULT%
