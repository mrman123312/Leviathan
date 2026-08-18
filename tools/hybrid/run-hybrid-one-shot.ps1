param(
  [Parameter(Mandatory=$true)][string]$Engine,
  [Parameter(Mandatory=$true)][string]$OpponentEngine,
  [int]$Threads = 8,
  [int]$Hash = 128,
  [int]$Games = 80,
  [int]$FastNodes = 50000,
  [int]$DeepNodes = 800000,
  [int]$ReplyNodes = 12000,
  [int]$OpponentLabelNodes = 50000,
  [int]$PonderMs = 2000,
  [int]$OwnMs = 250,
  [ValidateSet('auto','cuda','dml','cpu')][string]$Device = 'dml',
  [string]$Python = 'python',
  [string]$OutDir = 'local_results/hybrid/p18-one-shot'
)
$ErrorActionPreference='Stop'
function Run-Native([string]$Stage,[scriptblock]$Command){ & $Command; if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }

$root=Resolve-Path (Join-Path $PSScriptRoot '../..')
$out=Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $out | Out-Null

$all=Join-Path $out 'positions-all-v3.jsonl'
$trainPos=Join-Path $out 'positions-train-v3.jsonl'
$holdPos=Join-Path $out 'positions-holdout-v3.jsonl'
$trainRows=Join-Path $out 'train-v3.jsonl'
$holdRows=Join-Path $out 'prospective-v3.jsonl'
$replyTrain=Join-Path $out 'reply-pool-train-v4.jsonl'
$replyHold=Join-Path $out 'reply-pool-prospective-v4.jsonl'
$model=Join-Path $out 'p18.4.pt'
$metrics=Join-Path $out 'p18.4.metrics.json'
$warm=Join-Path $out 'warm-advantage-v4.json'

Write-Host '=== P18.4 accelerator preflight ===' -ForegroundColor Cyan
Run-Native 'NVIDIA preflight' { nvidia-smi }
if($Device -eq 'dml'){
  Run-Native 'DirectML PyTorch preflight' { & $Python -c "import json,torch,torch_directml; d=torch_directml.device(); x=torch.tensor([[1.,2.],[3.,4.]]).to(d); y=(x@x).cpu(); print(json.dumps({'torch':torch.__version__,'accelerator':'DirectML','device':str(d),'probe':y.tolist()},indent=2))" }
}else{
  Run-Native 'PyTorch accelerator preflight' { & $Python -c "import json,torch; dev='$Device'; ok=(dev=='cpu') or (dev=='auto') or (dev=='cuda' and torch.cuda.is_available()); print(json.dumps({'torch':torch.__version__,'requested':dev,'cuda_available':torch.cuda.is_available()},indent=2)); assert ok" }
}
Run-Native 'Advisor accelerator self-test' { & $Python (Join-Path $PSScriptRoot 'gpu_risk_model.py') --self-test --device $Device }

if(-not (Test-Path $all)){
  Write-Host '=== Generate game-grouped engine-distribution positions ===' -ForegroundColor Cyan
  Run-Native 'Position generation' { & $Python (Join-Path $PSScriptRoot 'generate_training_positions.py') --engine $OpponentEngine --output $all --games $Games --threads 1 --hash 32 --grouped-jsonl }
}else{
  $n=(Get-Content $all | Measure-Object -Line).Lines
  Write-Host "=== Reusing $n grouped positions ===" -ForegroundColor Green
}
if((-not (Test-Path $trainPos)) -or (-not (Test-Path $holdPos))){
  Run-Native 'Whole-game prospective split' { & $Python (Join-Path $PSScriptRoot 'split_positions.py') --input $all --train $trainPos --holdout $holdPos --holdout-frac 0.20 }
}else{
  $nt=(Get-Content $trainPos | Measure-Object -Line).Lines
  $nh=(Get-Content $holdPos | Measure-Object -Line).Lines
  Write-Host "=== Reusing frozen whole-game split: train=$nt holdout=$nh ===" -ForegroundColor Green
}

Write-Host '=== Mine/resume cheap 8-reply TRAIN pool ===' -ForegroundColor Cyan
Run-Native 'Train reply-pool mining' { & $Python (Join-Path $PSScriptRoot 'mine_reply_pool.py') --opponent-engine $OpponentEngine --positions $trainPos --output $replyTrain --reply-nodes $ReplyNodes --opponent-label-nodes $OpponentLabelNodes --multipv 8 --hash 32 }
Write-Host '=== Mine/resume cheap 8-reply PROSPECTIVE pool ===' -ForegroundColor Cyan
Run-Native 'Prospective reply-pool mining' { & $Python (Join-Path $PSScriptRoot 'mine_reply_pool.py') --opponent-engine $OpponentEngine --positions $holdPos --output $replyHold --reply-nodes $ReplyNodes --opponent-label-nodes $OpponentLabelNodes --multipv 8 --hash 32 }

Write-Host '=== Early untouched candidate-coverage gate ===' -ForegroundColor Cyan
& $Python (Join-Path $PSScriptRoot 'check_reply_pool.py') $replyHold --min-coverage 0.90
if($LASTEXITCODE-ne 0){
  Write-Warning 'MultiPV-8 shallow predictor does not cover the stronger opponent reply often enough. Stop before expensive deep-oracle mining.'
  exit $LASTEXITCODE
}

Write-Host '=== Mine/resume expensive TRAIN risk/regret labels (MultiPV 4 only) ===' -ForegroundColor Cyan
Run-Native 'Training risk-label mining' { & $Python (Join-Path $PSScriptRoot 'mine_finite_compute.py') --engine $Engine --opponent-engine $OpponentEngine --positions $trainPos --output $trainRows --reply-nodes $ReplyNodes --opponent-label-nodes $OpponentLabelNodes --fast-nodes $FastNodes --deep-nodes $DeepNodes --multipv 4 --threads $Threads --hash $Hash }

Write-Host '=== Mine/resume expensive PROSPECTIVE risk/regret labels (MultiPV 4 only) ===' -ForegroundColor Cyan
Run-Native 'Prospective risk-label mining' { & $Python (Join-Path $PSScriptRoot 'mine_finite_compute.py') --engine $Engine --opponent-engine $OpponentEngine --positions $holdPos --output $holdRows --reply-nodes $ReplyNodes --opponent-label-nodes $OpponentLabelNodes --fast-nodes $FastNodes --deep-nodes $DeepNodes --multipv 4 --threads $Threads --hash $Hash }

Write-Host '=== Train P18.4 game-leakage-safe / decision-correct three-head advisor ===' -ForegroundColor Cyan
& $Python (Join-Path $PSScriptRoot 'train_risk_model_v4.py') $trainRows $replyTrain --prospective $holdRows $replyHold --output $model --metrics-output $metrics --device $Device --hidden 48 --epochs 160 --patience 20 --min-reply-coverage 0.90
if($LASTEXITCODE-ne 0){
  Write-Warning 'Prospective P18.4 model gates failed. Checkpoint saved for research only.'
  exit $LASTEXITCODE
}

Write-Host '=== Prove correct ponder hits buy useful work ===' -ForegroundColor Cyan
& $Python (Join-Path $PSScriptRoot 'benchmark_warm_advantage.py') --engine $Engine --dataset $holdRows --output $warm --ponder-ms $PonderMs --own-ms $OwnMs --threads $Threads --hash $Hash --limit 48
if($LASTEXITCODE-ne 0){
  Write-Warning 'Warm-search advantage gate failed. Hybrid remains experimental.'
  exit $LASTEXITCODE
}

Write-Host '=== P18.4 checkpoint + coverage + warm-search gates PASSED ===' -ForegroundColor Green
Run-Native 'Checkpoint accelerator load' { & $Python (Join-Path $PSScriptRoot 'gpu_risk_model.py') --device $Device --checkpoint $model }
Write-Host "MODEL=$model"
Write-Host "METRICS=$metrics"
Write-Host "WARM_BENCH=$warm"
Write-Host 'Launch UCI proxy with:' -ForegroundColor Cyan
Write-Host "$Python tools/hybrid/leviathan_hybrid_uci_v2.py --engine `"$Engine`" --opponent-engine `"$OpponentEngine`" --model `"$model`" --gpu-device auto --threads $Threads --hash $Hash --max-scouts 4 --anneal-seconds 2.0 --min-final-scouts 2"
