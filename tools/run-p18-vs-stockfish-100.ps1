$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }
function Free-GB([string]$Path){
  $root=[IO.Path]::GetPathRoot((Resolve-Path $Path).Path);$name=$root.Substring(0,1);$d=Get-PSDrive -Name $name
  return [math]::Round($d.Free/1GB,2)
}

# Source commit containing: corrected regret decode, decision-level miner resume,
# and decisive no-GPU replay harness.
$RepoCommit='2d80c5178d56424fe09cc8204c633b230d828835'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$Work=Join-Path $Root 'p18.2-one-shot-work'
New-Item -ItemType Directory -Force -Path $Root | Out-Null

Write-Host '=== LEVIATHAN MATCH-FIRST: 100 GAMES BEFORE LONG MINING ===' -ForegroundColor Magenta
Write-Host "Pinned research source: $RepoCommit"

$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){ throw "No stockfish-baseline.exe found under $Root. Run the hardware baseline build first." }
$Stockfish=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'

Write-Host '=== SETUP ONLY: SYNC SOURCE / PRESERVE LOCAL RESULTS ===' -ForegroundColor Cyan
if(Test-Path (Join-Path $Work '.git')){
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host;Assert-LastExit 'git fetch pinned P18.4'
  git -C $Work reset --hard FETCH_HEAD | Out-Host;Assert-LastExit 'git reset pinned P18.4'
}else{
  if(Test-Path $Work){ Remove-Item -Recurse -Force $Work }
  git clone --no-checkout https://github.com/mrman123312/Leviathan.git $Work | Out-Host;Assert-LastExit 'git clone'
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host;Assert-LastExit 'git fetch pinned P18.4'
  git -C $Work checkout --detach FETCH_HEAD | Out-Host;Assert-LastExit 'git checkout pinned P18.4'
}

$oldCuda=Join-Path $Root 'p18-cuda-venv'
if(Test-Path $oldCuda){ Remove-Item -Recurse -Force $oldCuda -ErrorAction SilentlyContinue }
if(Test-Path $env:TEMP){
  Get-ChildItem $env:TEMP -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'pip-unpack-*' -or $_.Name -like 'pip-install-*' -or $_.Name -like 'pip-ephem-wheel-cache-*' -or $_.Name -like 'pip-build-env-*' } | ForEach-Object { try{Remove-Item -Recurse -Force $_.FullName -ErrorAction Stop}catch{} }
}

$bash='C:\msys64\usr\bin\bash.exe'
$compiler='C:\msys64\ucrt64\bin\x86_64-w64-mingw32-c++.exe'
if(-not(Test-Path $bash)){throw 'MSYS2 bash missing at C:\msys64\usr\bin\bash.exe'}
if(-not(Test-Path $compiler)){throw 'MSYS2 UCRT64 compiler missing at C:\msys64\ucrt64\bin\x86_64-w64-mingw32-c++.exe'}
$msysPath=$Work -replace '\\','/'
if($msysPath -match '^([A-Za-z]):/(.*)$'){$msysPath='/'+$matches[1].ToLower()+'/'+$matches[2]}

Write-Host '=== SETUP ONLY: VERIFY / BUILD P09 CPU CORE ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; command -v x86_64-w64-mingw32-c++ >/dev/null"
Assert-LastExit 'UCRT64 toolchain verification'
$P09=Join-Path $Work 'src\stockfish.exe'
$treeStamp=Join-Path $Root 'p18-p09-src-tree.txt'
$currentTree=(git -C $Work rev-parse HEAD:src).Trim();Assert-LastExit 'P09 source tree hash'
$priorTree=if(Test-Path $treeStamp){(Get-Content $treeStamp -Raw).Trim()}else{''}
if((Test-Path $P09) -and $priorTree -eq $currentTree){
  Write-Host 'Reusing already-built P09 static-race core.' -ForegroundColor Green
}else{
  & $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; cd '$msysPath/src' && make -j2 build ARCH=x86-64-avx2 COMP=mingw"
  Assert-LastExit 'P09 build'
  if(-not(Test-Path $P09)){throw "P09 binary missing after build: $P09"}
  Set-Content -Path $treeStamp -Value $currentTree -NoNewline
}

Write-Host '=== SETUP ONLY: VERIFY DIRECTML + PYTHON-CHESS ===' -ForegroundColor Cyan
$hostPython=(Get-Command python -ErrorAction Stop).Source
$Venv=Join-Path $Root 'p18-dml-venv'
$Py=Join-Path $Venv 'Scripts\python.exe'
$dmlReady=$false
if(Test-Path $Py){
  & $Py -c "import torch,torch_directml; d=torch_directml.device(); x=torch.tensor([2.]).to(d); assert float((x*x).sum().cpu())==4.0" 2>$null
  if($LASTEXITCODE-eq 0){$dmlReady=$true}
}
if(-not $dmlReady){
  if((Free-GB $Root) -lt 3.0){throw 'Need at least 3 GB free on C: to create the DirectML environment.'}
  if(Test-Path $Venv){Remove-Item -Recurse -Force $Venv}
  & $hostPython -m venv $Venv;Assert-LastExit 'DirectML venv creation'
  & $Py -m pip install --disable-pip-version-check --no-cache-dir torch-directml
  Assert-LastExit 'torch-directml install'
}
& $Py -c "import chess,chess.engine" 2>$null
if($LASTEXITCODE-ne 0){
  & $Py -m pip install --disable-pip-version-check --no-cache-dir python-chess
  Assert-LastExit 'python-chess install'
}
& $Py -c "import json,torch,torch_directml,chess; d=torch_directml.device(); x=torch.tensor([[1.,2.],[3.,4.]]).to(d); print(json.dumps({'torch':torch.__version__,'python_chess':chess.__version__,'accelerator':'DirectML','device':str(d),'probe':(x@x).cpu().tolist()},indent=2))"
Assert-LastExit 'DirectML verification'

$ResultDir=Join-Path $Work 'local_results\hybrid\p18-one-shot'
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$FinalModel=Join-Path $ResultDir 'p18.4.pt'
$FinalMetrics=Join-Path $ResultDir 'p18.4.metrics.json'
$FinalWarm=Join-Path $ResultDir 'warm-advantage-v4.json'
$ProvisionalModel=Join-Path $ResultDir 'p18.4-match-first.pt'
$ProvisionalMetrics=Join-Path $ResultDir 'p18.4-match-first.metrics.json'
$TrainRows=Join-Path $ResultDir 'train-v3.jsonl'
$ReplyTrain=Join-Path $ResultDir 'reply-pool-train-v4.jsonl'

function Test-P18Promotion {
  if((-not(Test-Path $FinalModel)) -or (-not(Test-Path $FinalMetrics)) -or (-not(Test-Path $FinalWarm))){return $false}
  try{
    $m=Get-Content $FinalMetrics -Raw | ConvertFrom-Json
    $w=Get-Content $FinalWarm -Raw | ConvertFrom-Json
    return ([bool]$m.promote -and [bool]$w.summary.pass)
  }catch{return $false}
}

# Pick the strongest already-existing checkpoint first. If full P18.4 is not there yet,
# train a clearly-labelled provisional checkpoint only from data already on disk.
$MatchModel=$null
$MatchModelKind=$null
if(Test-P18Promotion){
  $MatchModel=$FinalModel;$MatchModelKind='promoted-p18.4'
}elseif(Test-Path $FinalModel){
  $MatchModel=$FinalModel;$MatchModelKind='existing-unpromoted-p18.4'
}elseif(Test-Path $ProvisionalModel){
  $MatchModel=$ProvisionalModel;$MatchModelKind='existing-provisional'
}else{
  if((-not(Test-Path $ReplyTrain)) -and (-not(Test-Path $TrainRows))){
    throw 'No trained advisor and no partial training data exist yet. Need at least some saved mining rows before a real GPU/model match can be run.'
  }
  $datasets=@()
  if(Test-Path $TrainRows){$datasets += $TrainRows}
  if(Test-Path $ReplyTrain){$datasets += $ReplyTrain}
  Write-Host '=== MINIMUM PREREQUISITE: TRAIN PROVISIONAL ADVISOR FROM DATA ALREADY ON DISK ===' -ForegroundColor Yellow
  Write-Host 'This is NOT the final/gated P18.4 model. It exists only so the requested GPU-vs-no-GPU match is real.' -ForegroundColor Yellow
  $TrainerPy=Join-Path $Work 'tools\hybrid\train_risk_model_v4.py'
  & $Py $TrainerPy @datasets --output $ProvisionalModel --metrics-output $ProvisionalMetrics --device dml --hidden 48 --epochs 80 --patience 12 --min-risk-auc 0 --min-top20-regret-capture 0 --min-reply-top1-gain -1 --min-reply-coverage 0
  $trainExit=$LASTEXITCODE
  if(-not(Test-Path $ProvisionalModel)){throw "Provisional advisor training did not produce a checkpoint (exit=$trainExit)."}
  if($trainExit-ne 0){Write-Warning "Provisional model did not satisfy its research metrics (exit=$trainExit), but checkpoint exists and will be used for this exploratory match as requested."}
  $MatchModel=$ProvisionalModel;$MatchModelKind='new-provisional-from-partial-data'
}

Write-Host "=== VERIFY MATCH MODEL ON DIRECTML: $MatchModelKind ===" -ForegroundColor Cyan
& $Py (Join-Path $Work 'tools\hybrid\gpu_risk_model.py') --device dml --checkpoint $MatchModel
Assert-LastExit 'match checkpoint DirectML load'

# FIRST SUBSTANTIVE EXPERIMENT: the 100-game match. Every decisive game is immediately
# replayed with the same P18 architecture but learned GPU/model advisor disabled.
Write-Host '=== FIRST EXPERIMENT: START 100-GAME LEVIATHAN CPU+GPU VS STOCKFISH ===' -ForegroundColor Magenta
Write-Host "Advisor used: $MatchModelKind"
Write-Host 'WIN or LOSS => immediate same-opening/same-color no-GPU replay. DRAW => no replay.' -ForegroundColor Yellow
Write-Host 'Both variants keep P18 multi-ponder; the ablation disables the learned GPU/model advisor only.'
$MatchOut=Join-Path $Work 'local_results\hybrid\p18-vs-stockfish-100-match-first'
$Harness=Join-Path $Work 'tools\hybrid\run_p18_vs_stockfish_100.py'
$HybridScript=Join-Path $Work 'tools\hybrid\leviathan_hybrid_uci_v2.py'
& $Py $Harness --engine $P09 --opponent-engine $Stockfish --model $MatchModel --hybrid-script $HybridScript --out-dir $MatchOut --games 100 --movetime-ms 500 --max-plies 240 --threads 0 --hash 128 --max-scouts 4 --reply-nodes 12000 --anneal-seconds 0.15 --min-final-scouts 2 --opening-plies 10 --opening-nodes 1500 --seed 20260818
Assert-LastExit 'MATCH-FIRST 100-game P18 vs Stockfish + decisive no-GPU ablations'
Write-Host '=== MATCH-FIRST 100 GAMES COMPLETE ===' -ForegroundColor Green
Write-Host "Results root: $MatchOut" -ForegroundColor Green

# Only AFTER the requested match is complete do we resume/finish the expensive final campaign.
if(-not(Test-P18Promotion)){
  Write-Host '=== NOW RESUME/FINISH FULL P18.4 MINING + HOLDOUT + WARM GATES ===' -ForegroundColor Cyan
  $FullTrainer=Join-Path $Work 'tools\hybrid\run-hybrid-one-shot.ps1'
  & $FullTrainer -Engine $P09 -OpponentEngine $Stockfish -Threads 8 -Hash 128 -Games 80 -Python $Py -Device dml -OutDir 'local_results/hybrid/p18-one-shot'
  if($LASTEXITCODE-ne 0){
    Write-Warning 'Full P18.4 campaign stopped or failed a promotion gate. The match-first results remain saved.'
    exit $LASTEXITCODE
  }
}

if(Test-P18Promotion){
  Write-Host '=== FINAL P18.4 IS PROMOTED ===' -ForegroundColor Green
  Write-Host 'The requested 100-game match already ran first. Re-run this same one-liner later if you want the now-final model to receive its own new 100-game match.' -ForegroundColor Green
}else{
  Write-Warning 'Full P18.4 did not finish promoted, but the requested match-first experiment is saved.'
}
