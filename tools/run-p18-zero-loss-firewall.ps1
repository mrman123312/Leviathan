$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }

$RepoCommit='041ebc0822e6ddafc67184e3bb9fe17a3d1afff5'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$Work=Join-Path $Root 'p18.2-one-shot-work'
$Venv=Join-Path $Root 'p18-dml-venv'
$Py=Join-Path $Venv 'Scripts\python.exe'

Write-Host '=== P18.8 ZERO-LOSS FIREWALL ===' -ForegroundColor Magenta
Write-Host 'The provisional learned advisor is BENCHED from authority.' -ForegroundColor Yellow
Write-Host 'Known losses are permanent sentinels. Any new loss rejects the candidate immediately.' -ForegroundColor Yellow
Write-Host 'A loss is never rewritten, retried away, or hidden.' -ForegroundColor Yellow
Write-Host "Pinned research source: $RepoCommit"

$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){ throw "No stockfish-baseline.exe found under $Root." }
$Stockfish=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'

Write-Host '=== SYNC P18.8 SOURCE / PRESERVE LOCAL RESULTS ===' -ForegroundColor Cyan
if(Test-Path (Join-Path $Work '.git')){
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P18.8'
  git -C $Work reset --hard FETCH_HEAD | Out-Host; Assert-LastExit 'git reset P18.8'
}else{
  git clone --no-checkout https://github.com/mrman123312/Leviathan.git $Work | Out-Host; Assert-LastExit 'git clone'
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P18.8'
  git -C $Work checkout --detach FETCH_HEAD | Out-Host; Assert-LastExit 'git checkout P18.8'
}

if(-not(Test-Path $Py)){ throw "DirectML Python environment missing: $Py" }
& $Py -c "import torch,torch_directml,chess; d=torch_directml.device(); x=torch.tensor([2.]).to(d); assert float((x*x).sum().cpu())==4.; print('DirectML OK:',d,'python-chess',chess.__version__)"
Assert-LastExit 'DirectML/python-chess preflight'

$P09=Join-Path $Work 'src\stockfish.exe'
if(-not(Test-Path $P09)){
  $bash='C:\msys64\usr\bin\bash.exe'
  if(-not(Test-Path $bash)){throw 'P09 binary absent and MSYS2 bash missing.'}
  $msysPath=$Work -replace '\\','/'
  if($msysPath -match '^([A-Za-z]):/(.*)$'){$msysPath='/'+$matches[1].ToLower()+'/'+$matches[2]}
  & $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; cd '$msysPath/src' && make -j2 build ARCH=x86-64-avx2 COMP=mingw"
  Assert-LastExit 'P09 build'
}

$ResultDir=Join-Path $Work 'local_results\hybrid\p18-one-shot'
$FinalModel=Join-Path $ResultDir 'p18.4.pt'
$ProvisionalModel=Join-Path $ResultDir 'p18.4-match-first.pt'
if(Test-Path $FinalModel){$Model=$FinalModel;$ModelKind='existing-p18.4'}
elseif(Test-Path $ProvisionalModel){$Model=$ProvisionalModel;$ModelKind='existing-provisional'}
else{throw 'No P18 advisor checkpoint found.'}

Write-Host "=== INSPECT ADVISOR: $ModelKind ===" -ForegroundColor Cyan
& $Py (Join-Path $Work 'tools\hybrid\gpu_risk_model.py') --device dml --checkpoint $Model
Assert-LastExit 'DirectML checkpoint inspection'
Write-Host 'NOTE: model inference is telemetry-only in this gate. Heuristic counterfactual remains authoritative.' -ForegroundColor Yellow

Write-Host '=== KNOWN-LOSS SENTINELS + FRESH ZERO-LOSS MATCH ===' -ForegroundColor Magenta
Write-Host 'Sentinels: P18.7 game 24 and game 27, three repetitions each.' -ForegroundColor Yellow
Write-Host 'If a sentinel loses: ABORT. If any fresh game loses: replay without model, save counterexample, ABORT.' -ForegroundColor Yellow

$Harness=Join-Path $Work 'tools\hybrid\run_p18_zero_loss_gate.py'
$HybridScript=Join-Path $Work 'tools\hybrid\leviathan_hybrid_uci_v6.py'
$MatchOut=Join-Path $Work 'local_results\hybrid\p18-zero-loss-firewall-v6'
& $Py $Harness --engine $P09 --opponent-engine $Stockfish --model $Model --hybrid-script $HybridScript --out-dir $MatchOut --games 100 --movetime-ms 500 --max-plies 240 --threads 0 --hash 128 --max-scouts 4 --reply-nodes 12000 --anneal-seconds 0 --min-final-scouts 2 --opening-plies 10 --opening-nodes 1500 --seed 20260818 --sentinel-repeats 3
Assert-LastExit 'P18.8 zero-loss firewall'

Write-Host '=== ZERO-LOSS FIREWALL PASSED ===' -ForegroundColor Green
Write-Host 'The provisional learned model caused no authoritative branch/allocation decisions in this test.' -ForegroundColor Green
Write-Host "Results: $MatchOut" -ForegroundColor Green

Write-Host '=== NOW RESUME FULL RISK/REGRET TRAINING ===' -ForegroundColor Cyan
Write-Host 'Future learned authority remains locked until real risk/regret validation exists and the loss sentinels pass.' -ForegroundColor Yellow
$FullTrainer=Join-Path $Work 'tools\hybrid\run-hybrid-one-shot.ps1'
& $FullTrainer -Engine $P09 -OpponentEngine $Stockfish -Threads 8 -Hash 128 -Games 80 -Python $Py -Device cpu -OutDir 'local_results/hybrid/p18-one-shot'
if($LASTEXITCODE-ne 0){ Write-Warning 'Full training/gates stopped. Zero-loss firewall results remain saved.'; exit $LASTEXITCODE }

Write-Host '=== FULL TRAINING RESUMED/COMPLETED ===' -ForegroundColor Green
Write-Host 'Do NOT grant learned advisor authority merely because training completed; it must pass qualification + loss sentinels first.' -ForegroundColor Yellow
