$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }
function Free-GB([string]$Path){
  $root=[IO.Path]::GetPathRoot((Resolve-Path $Path).Path);$name=$root.Substring(0,1);$d=Get-PSDrive -Name $name
  return [math]::Round($d.Free/1GB,2)
}

$RepoCommit='2d80c5178d56424fe09cc8204c633b230d828835'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$Work=Join-Path $Root 'p18.2-one-shot-work'
New-Item -ItemType Directory -Force -Path $Root | Out-Null

Write-Host '=== P18.4 FULL CPU+GPU VS STOCKFISH / 100 GAMES + DECISIVE NO-GPU REPLAYS ===' -ForegroundColor Magenta
Write-Host "Pinned research commit: $RepoCommit"

$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){ throw "No stockfish-baseline.exe found under $Root. Run the hardware baseline build first." }
$Stockfish=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'

Write-Host '=== SYNC PINNED P18.4 SOURCE (PRESERVE LOCAL RESULTS) ===' -ForegroundColor Cyan
if(Test-Path (Join-Path $Work '.git')){
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host;Assert-LastExit 'git fetch pinned P18.4'
  git -C $Work reset --hard FETCH_HEAD | Out-Host;Assert-LastExit 'git reset pinned P18.4'
}else{
  if(Test-Path $Work){ Remove-Item -Recurse -Force $Work }
  git clone --no-checkout https://github.com/mrman123312/Leviathan.git $Work | Out-Host;Assert-LastExit 'git clone'
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host;Assert-LastExit 'git fetch pinned P18.4'
  git -C $Work checkout --detach FETCH_HEAD | Out-Host;Assert-LastExit 'git checkout pinned P18.4'
}

Write-Host '=== CLEAN ONLY OBSOLETE P18 INSTALL TEMP ===' -ForegroundColor Cyan
$oldCuda=Join-Path $Root 'p18-cuda-venv'
if(Test-Path $oldCuda){ Remove-Item -Recurse -Force $oldCuda -ErrorAction SilentlyContinue }
if(Test-Path $env:TEMP){
  Get-ChildItem $env:TEMP -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'pip-unpack-*' -or $_.Name -like 'pip-install-*' -or $_.Name -like 'pip-ephem-wheel-cache-*' -or $_.Name -like 'pip-build-env-*' } | ForEach-Object { try{Remove-Item -Recurse -Force $_.FullName -ErrorAction Stop}catch{} }
}
Write-Host "Free disk: $(Free-GB $Root) GB"

$bash='C:\msys64\usr\bin\bash.exe'
$compiler='C:\msys64\ucrt64\bin\x86_64-w64-mingw32-c++.exe'
if(-not(Test-Path $bash)){throw 'MSYS2 bash missing at C:\msys64\usr\bin\bash.exe'}
if(-not(Test-Path $compiler)){throw 'MSYS2 UCRT64 compiler missing at C:\msys64\ucrt64\bin\x86_64-w64-mingw32-c++.exe'}
$msysPath=$Work -replace '\\','/'
if($msysPath -match '^([A-Za-z]):/(.*)$'){$msysPath='/'+$matches[1].ToLower()+'/'+$matches[2]}

Write-Host '=== VERIFY / BUILD P09 CPU CORE ===' -ForegroundColor Cyan
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

Write-Host '=== VERIFY / PREPARE DIRECTML GPU ENVIRONMENT ===' -ForegroundColor Cyan
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
& $Py -c "import json,torch,torch_directml; d=torch_directml.device(); x=torch.tensor([[1.,2.],[3.,4.]]).to(d); print(json.dumps({'torch':torch.__version__,'accelerator':'DirectML','device':str(d),'probe':(x@x).cpu().tolist()},indent=2))"
Assert-LastExit 'DirectML verification'

Write-Host '=== VERIFY PYTHON-CHESS MATCH HARNESS DEPENDENCY ===' -ForegroundColor Cyan
& $Py -c "import chess,chess.engine; print('python-chess',chess.__version__)" 2>$null
if($LASTEXITCODE-ne 0){
  & $Py -m pip install --disable-pip-version-check --no-cache-dir python-chess
  Assert-LastExit 'python-chess install'
}

$ResultDir=Join-Path $Work 'local_results\hybrid\p18-one-shot'
$Model=Join-Path $ResultDir 'p18.4.pt'
$Metrics=Join-Path $ResultDir 'p18.4.metrics.json'
$Warm=Join-Path $ResultDir 'warm-advantage-v4.json'

function Test-P18Promotion {
  if((-not(Test-Path $Model)) -or (-not(Test-Path $Metrics)) -or (-not(Test-Path $Warm))){return $false}
  try{
    $m=Get-Content $Metrics -Raw | ConvertFrom-Json
    $w=Get-Content $Warm -Raw | ConvertFrom-Json
    return ([bool]$m.promote -and [bool]$w.summary.pass)
  }catch{return $false}
}

if(-not(Test-P18Promotion)){
  Write-Host '=== P18.4 IS NOT PROMOTED YET: RESUME/FINISH TRAINING + HOLDOUT + WARM GATES ===' -ForegroundColor Yellow
  $Trainer=Join-Path $Work 'tools\hybrid\run-hybrid-one-shot.ps1'
  & $Trainer -Engine $P09 -OpponentEngine $Stockfish -Threads 8 -Hash 128 -Games 80 -Python $Py -Device dml -OutDir 'local_results/hybrid/p18-one-shot'
  if($LASTEXITCODE-ne 0){throw 'P18.4 failed a promotion gate. The 100-game match was intentionally NOT started.'}
}
if(-not(Test-P18Promotion)){throw 'P18.4 promotion artifacts are still missing or did not pass; refusing to benchmark heuristic fallback.'}

Write-Host '=== VERIFY PROMOTED MODEL LOADS ON DIRECTML ===' -ForegroundColor Cyan
& $Py (Join-Path $Work 'tools\hybrid\gpu_risk_model.py') --device dml --checkpoint $Model
Assert-LastExit 'P18.4 DirectML checkpoint load'

Write-Host '=== START 100-GAME FAIR PONDER MATCH + DECISIVE NO-GPU ABLATIONS ===' -ForegroundColor Magenta
Write-Host 'Every GPU-enabled WIN or LOSS is replayed from the exact same opening/color with the same P18 architecture but GPU/model disabled.'
Write-Host 'Draws are not replayed. Both variants face the same Stockfish with equal CPU/hash/movetime and pondering.'
$MatchOut=Join-Path $Work 'local_results\hybrid\p18-vs-stockfish-100'
$Harness=Join-Path $Work 'tools\hybrid\run_p18_vs_stockfish_100.py'
$HybridScript=Join-Path $Work 'tools\hybrid\leviathan_hybrid_uci_v2.py'
& $Py $Harness --engine $P09 --opponent-engine $Stockfish --model $Model --hybrid-script $HybridScript --out-dir $MatchOut --games 100 --movetime-ms 500 --max-plies 240 --threads 0 --hash 128 --max-scouts 4 --reply-nodes 12000 --anneal-seconds 0.15 --min-final-scouts 2 --opening-plies 10 --opening-nodes 1500 --seed 20260818
Assert-LastExit '100-game P18.4 vs Stockfish match + decisive no-GPU ablations'

Write-Host "`n=== DONE: P18.4 CPU+GPU VS STOCKFISH 100 GAMES + DECISIVE NO-GPU ABLATIONS ===" -ForegroundColor Green
Write-Host "Results root: $MatchOut" -ForegroundColor Green
