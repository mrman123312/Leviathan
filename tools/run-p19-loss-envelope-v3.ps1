$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }

$RepoCommit='d639e16d6eeddf3cb5e1df6b9e97878881e5a1ad'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$Work=Join-Path $Root 'p19-loss-envelope-work'
$Venv=Join-Path $Root 'p18-dml-venv'
$Py=Join-Path $Venv 'Scripts\python.exe'
if(-not(Test-Path $Py)){$Py=(Get-Command python -ErrorAction Stop).Source}

Write-Host '=== P19.2 LOSS-ENVELOPE SURVIVAL ===' -ForegroundColor Magenta
Write-Host 'P19.1 rejected an approximately -0.4 pawn drawable-looking position because its absolute -20cp floor was the wrong invariant.' -ForegroundColor Red
Write-Host 'New invariant: do not materially increase worst estimated LOSS probability across an all-reply hostile envelope.' -ForegroundColor Yellow
Write-Host 'Mate/decisive loss is still a hard veto. Immediate legal draw is a hard survival lock.' -ForegroundColor Yellow
Write-Host 'P09 + frozen Stockfish critics: 1 thread, fixed nodes. No learned GPU authority. No master merge.' -ForegroundColor Yellow
Write-Host "Pinned research source: $RepoCommit"

New-Item -ItemType Directory -Force -Path $Root | Out-Null
if(Test-Path (Join-Path $Work '.git')){
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P19.2'
  git -C $Work reset --hard FETCH_HEAD | Out-Host; Assert-LastExit 'git reset P19.2'
}else{
  git clone --no-checkout https://github.com/mrman123312/Leviathan.git $Work | Out-Host; Assert-LastExit 'git clone P19.2'
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P19.2'
  git -C $Work checkout --detach FETCH_HEAD | Out-Host; Assert-LastExit 'git checkout P19.2'
}

& $Py -c "import chess; print('python-chess',chess.__version__)"
if($LASTEXITCODE-ne 0){ & $Py -m pip install chess; Assert-LastExit 'install python-chess' }

$P09=Join-Path $Work 'src\stockfish.exe'
$bash='C:\msys64\usr\bin\bash.exe'
if(-not(Test-Path $bash)){throw 'MSYS2 bash missing at C:\msys64\usr\bin\bash.exe'}
$msysPath=$Work -replace '\\','/'
if($msysPath -match '^([A-Za-z]):/(.*)$'){$msysPath='/'+$matches[1].ToLower()+'/'+$matches[2]}
Write-Host '=== BUILD P19.2 FROM P09 STATIC-RACE CORE ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; cd '$msysPath/src' && make -j2 build ARCH=x86-64-avx2 COMP=mingw"
Assert-LastExit 'P19.2/P09 build'
if(-not(Test-Path $P09)){throw "P19.2 engine binary missing after build: $P09"}

$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){throw 'No stockfish-baseline.exe found under LeviathanHardwareResults.'}
$Stockfish=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'
Write-Host "Frozen Stockfish: $Stockfish" -ForegroundColor DarkGray

$Harness=Join-Path $Work 'tools\hybrid\run_p19_loss_envelope_v3.py'
$Out=Join-Path $Work 'local_results\p19-loss-envelope-v3'
Write-Host '=== 20 HISTORICAL/MICRO SENTINELS, THEN FRESH 100-GAME GATE ===' -ForegroundColor Magenta
Write-Host 'Each candidate: WDL baseline -> all opponent replies -> deepest 6 hostile replies -> relative loss envelope.' -ForegroundColor DarkGray
Write-Host 'Normal tolerance: at most +15/1000 worst loss probability vs the current-position envelope.' -ForegroundColor DarkGray
Write-Host 'If normal candidates fail, panic mode enumerates every Leviathan move and every opponent reply at larger node budgets.' -ForegroundColor DarkGray

& $Py $Harness `
  --engine $P09 `
  --opponent-engine $Stockfish `
  --out-dir $Out `
  --games 100 `
  --opponent-threads 6 `
  --hash 128 `
  --opponent-movetime-ms 500 `
  --max-plies 240 `
  --opening-plies 10 `
  --opening-nodes 1500 `
  --seed 20260818 `
  --sentinel-repeats 5 `
  --baseline-nodes 120000 `
  --root-nodes 40000 `
  --reply-nodes 30000 `
  --deep-reply-nodes 120000 `
  --panic-root-nodes 180000 `
  --panic-reply-nodes 80000 `
  --panic-deep-nodes 300000 `
  --candidate-mpv 8 `
  --dangerous-replies 6 `
  --loss-delta-pm 15 `
  --max-loss-pm 400 `
  --disagreement-pm 80

$code=$LASTEXITCODE
if($code-eq 21){throw 'P19.2 REJECTED: a historical/micro sentinel produced an actual Leviathan loss.'}
if($code-eq 31){throw 'P19.2 REJECTED: a fresh Stockfish game produced an actual Leviathan loss.'}
if($code-eq 41){throw 'P19.2 NOT CERTIFIED: a sentinel exhausted the all-reply loss-envelope search without a safe move.'}
if($code-eq 42){throw 'P19.2 NOT CERTIFIED: a fresh game exhausted the all-reply loss-envelope search without a safe move.'}
Assert-LastExit 'P19.2 loss-envelope gate'

Write-Host '=== P19.2 ZERO-LOSS GATE PASSED ===' -ForegroundColor Green
Write-Host 'This is empirical survival evidence over the tested adversarial domain, not a mathematical solution of chess.' -ForegroundColor Yellow
Write-Host 'Next: adversarial disagreement/loss mining, sharper openings, deeper critics, then compress compute cost.' -ForegroundColor Yellow
Write-Host "Results: $Out" -ForegroundColor Green
