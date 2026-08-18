param(
    [Parameter(Mandatory=$true)][string]$Engine,
    [string]$OpponentEngine = "",
    [int]$Threads = 8,
    [int]$Hash = 128,
    [double]$PonderSeconds = 3.0,
    [string]$Positions = "",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "../..")
$out = Join-Path $root "local_results/hybrid"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Write-Host "=== Leviathan Hybrid RTX preflight ==="
& nvidia-smi
& $Python -c "import json,platform; import torch; print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2)); assert torch.cuda.is_available()"
& $Python (Join-Path $PSScriptRoot "gpu_risk_model.py") --self-test --device cuda
$smokeArgs = @((Join-Path $PSScriptRoot "real_hybrid_smoke.py"),"--engine",$Engine,"--device","cuda","--threads","$Threads","--hash","$Hash","--ponder-seconds","$PonderSeconds","--log",(Join-Path $out "real-smoke.jsonl"))
if ($OpponentEngine -ne "") { $smokeArgs += @("--opponent-engine",$OpponentEngine) }
& $Python @smokeArgs
if ($Positions -ne "") {
    $risk = Join-Path $out "finite-compute-risk.jsonl"
    $mineArgs = @((Join-Path $PSScriptRoot "mine_finite_compute.py"),"--engine",$Engine,"--positions",$Positions,"--output",$risk,"--reply-nodes","12000","--fast-nodes","50000","--deep-nodes","800000","--multipv","4","--threads","$Threads","--hash","$Hash")
    if ($OpponentEngine -ne "") { $mineArgs += @("--opponent-engine",$OpponentEngine) }
    & $Python @mineArgs
    Write-Host "Risk rows: $risk"
}
Write-Host "=== Hybrid smoke complete ==="
Write-Host "For real matches, launch leviathan_hybrid_uci.py in the UCI GUI and ENABLE Ponder."
