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
  [string]$Python = "python",
  [string]$OutDir = "local_results/hybrid/p18-one-shot"
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "../..")
$out = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $out | Out-Null
$all = Join-Path $out "positions-all.txt"; $trainPos = Join-Path $out "positions-train.txt"; $holdPos = Join-Path $out "positions-holdout.txt"
$trainRows = Join-Path $out "train.jsonl"; $holdRows = Join-Path $out "prospective.jsonl"; $model = Join-Path $out "p18.2.pt"; $metrics = Join-Path $out "p18.2.metrics.json"; $warm = Join-Path $out "warm-advantage.json"
Write-Host "=== P18.2 one-shot preflight ==="
& nvidia-smi
& $Python -c "import torch, json; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2)); assert torch.cuda.is_available()"
& $Python (Join-Path $PSScriptRoot "gpu_risk_model.py") --self-test --device cuda
Write-Host "=== Generate diverse engine-distribution positions ==="
& $Python (Join-Path $PSScriptRoot "generate_training_positions.py") --engine $OpponentEngine --output $all --games $Games --threads 1 --hash 32
& $Python (Join-Path $PSScriptRoot "split_positions.py") --input $all --train $trainPos --holdout $holdPos --holdout-frac 0.20
Write-Host "=== Mine train labels (shallow reply probe vs stronger opponent truth + P09 risk/regret) ==="
& $Python (Join-Path $PSScriptRoot "mine_finite_compute.py") --engine $Engine --opponent-engine $OpponentEngine --positions $trainPos --output $trainRows --reply-nodes $ReplyNodes --opponent-label-nodes $OpponentLabelNodes --fast-nodes $FastNodes --deep-nodes $DeepNodes --multipv 4 --threads $Threads --hash $Hash
Write-Host "=== Mine untouched prospective holdout ==="
& $Python (Join-Path $PSScriptRoot "mine_finite_compute.py") --engine $Engine --opponent-engine $OpponentEngine --positions $holdPos --output $holdRows --reply-nodes $ReplyNodes --opponent-label-nodes $OpponentLabelNodes --fast-nodes $FastNodes --deep-nodes $DeepNodes --multipv 4 --threads $Threads --hash $Hash
Write-Host "=== Train leakage-resistant three-head advisor ==="
& $Python (Join-Path $PSScriptRoot "train_risk_model.py") $trainRows --prospective $holdRows --output $model --metrics-output $metrics --device cuda --hidden 48 --epochs 160 --patience 20
if ($LASTEXITCODE -ne 0) { Write-Warning "Promotion gates failed. Checkpoint was saved for research but MUST NOT be used as champion."; exit $LASTEXITCODE }
Write-Host "=== Prove correct ponder hits buy useful work ==="
& $Python (Join-Path $PSScriptRoot "benchmark_warm_advantage.py") --engine $Engine --dataset $holdRows --output $warm --ponder-ms $PonderMs --own-ms $OwnMs --threads $Threads --hash $Hash --limit 48
if ($LASTEXITCODE -ne 0) { Write-Warning "Warm-search advantage gate failed. Hybrid remains experimental."; exit $LASTEXITCODE }
Write-Host "=== Checkpoint and warm-search gates passed ==="
& $Python (Join-Path $PSScriptRoot "gpu_risk_model.py") --device cuda --checkpoint $model
Write-Host "MODEL=$model"
Write-Host "METRICS=$metrics"
Write-Host "WARM_BENCH=$warm"
Write-Host "Launch UCI proxy with:"
Write-Host "$Python tools/hybrid/leviathan_hybrid_uci_v2.py --engine `"$Engine`" --opponent-engine `"$OpponentEngine`" --model `"$model`" --gpu-device cuda --threads $Threads --hash $Hash --max-scouts 4 --anneal-seconds 2.0 --min-final-scouts 2"
