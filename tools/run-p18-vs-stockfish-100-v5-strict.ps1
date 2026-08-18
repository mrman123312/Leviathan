$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }

# P18.7 source contains:
# - V3 race-safe ponder cancellation
# - V4 atomic bestmove handoff
# - V5 dedicated speculative scouts: foreground primary is never in ponder pool
# - strict exact-ply diagnostics + automatic proxy log tail
$RepoCommit='a88b228d080858fa34cf87f29970c77969d3b3f8'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$Work=Join-Path $Root 'p18.2-one-shot-work'
$Venv=Join-Path $Root 'p18-dml-venv'
$Py=Join-Path $Venv 'Scripts\python.exe'

Write-Host '=== P18.7 STRICT MATCH-FIRST / DEDICATED SPECULATIVE SCOUTS ===' -ForegroundColor Magenta
Write-Host 'Foreground engine is never part of the ponder scout pool. No annealing. No silent retries.' -ForegroundColor Yellow
Write-Host 'Any remaining fault prints exact ply/FEN plus the last hybrid proxy events.' -ForegroundColor Yellow
Write-Host "Pinned research source: $RepoCommit"

$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){ throw "No stockfish-baseline.exe found under $Root." }
$Stockfish=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'

Write-Host '=== SYNC P18.7 SOURCE / PRESERVE LOCAL RESULTS ===' -ForegroundColor Cyan
if(Test-Path (Join-Path $Work '.git')){
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P18.7'
  git -C $Work reset --hard FETCH_HEAD | Out-Host; Assert-LastExit 'git reset P18.7'
}else{
  git clone --no-checkout https://github.com/mrman123312/Leviathan.git $Work | Out-Host; Assert-LastExit 'git clone'
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P18.7'
  git -C $Work checkout --detach FETCH_HEAD | Out-Host; Assert-LastExit 'git checkout P18.7'
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

Write-Host "=== VERIFY GPU MODEL: $ModelKind ===" -ForegroundColor Cyan
& $Py (Join-Path $Work 'tools\hybrid\gpu_risk_model.py') --device dml --checkpoint $Model
Assert-LastExit 'DirectML checkpoint load'

Write-Host '=== START FRESH P18.7 STRICT 100-GAME MATCH ===' -ForegroundColor Magenta
Write-Host 'DRAW => next game. WIN/LOSS => immediate same-opening/same-color learned-advisor-OFF replay.' -ForegroundColor Yellow
Write-Host 'GPU-off replay keeps the same dedicated-scout architecture; only learned advisor/model inference is removed.' -ForegroundColor Yellow

$Harness=Join-Path $Work 'tools\hybrid\run_p18_vs_stockfish_100_strict_v5.py'
$HybridScript=Join-Path $Work 'tools\hybrid\leviathan_hybrid_uci_v5.py'
$MatchOut=Join-Path $Work 'local_results\hybrid\p18-vs-stockfish-100-match-first-v5-strict'
& $Py $Harness --engine $P09 --opponent-engine $Stockfish --model $Model --hybrid-script $HybridScript --out-dir $MatchOut --games 100 --movetime-ms 500 --max-plies 240 --threads 0 --hash 128 --max-scouts 4 --reply-nodes 12000 --anneal-seconds 0 --min-final-scouts 2 --opening-plies 10 --opening-nodes 1500 --seed 20260818
Assert-LastExit 'P18.7 strict 100-game match + decisive no-GPU ablations'

Write-Host '=== P18.7 STRICT 100-GAME MATCH COMPLETE ===' -ForegroundColor Green
Write-Host "Results: $MatchOut" -ForegroundColor Green

Write-Host '=== NOW RESUME FULL EXPENSIVE P18.4 TRAINING CAMPAIGN ===' -ForegroundColor Cyan
$FullTrainer=Join-Path $Work 'tools\hybrid\run-hybrid-one-shot.ps1'
& $FullTrainer -Engine $P09 -OpponentEngine $Stockfish -Threads 8 -Hash 128 -Games 80 -Python $Py -Device cpu -OutDir 'local_results/hybrid/p18-one-shot'
if($LASTEXITCODE-ne 0){ Write-Warning 'Full training/gates stopped after the match. Strict match results remain saved.'; exit $LASTEXITCODE }
