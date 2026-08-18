$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }

$RepoCommit='b46e4a0cb77f4d284f28dcf997e119179b203ee8'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$Work=Join-Path $Root 'p19-certified-survival-work'
$Venv=Join-Path $Root 'p18-dml-venv'
$Py=Join-Path $Venv 'Scripts\python.exe'
if(-not(Test-Path $Py)){$Py=(Get-Command python -ErrorAction Stop).Source}

Write-Host '=== P19.1 CERTIFIED SURVIVAL / HETEROGENEOUS VETO ===' -ForegroundColor Magenta
Write-Host 'P19-v1 is rejected. Same sentinel drew once, then lost by checkmate.' -ForegroundColor Red
Write-Host 'No time-based Leviathan selector. P09 + frozen Stockfish are fixed-node, 1-thread critics.' -ForegroundColor Yellow
Write-Host 'If no move clears the certificate, the candidate FAILS UNCERTIFIED instead of gambling.' -ForegroundColor Yellow
Write-Host 'No learned GPU authority. No master merge.' -ForegroundColor Yellow
Write-Host "Pinned research source: $RepoCommit"

New-Item -ItemType Directory -Force -Path $Root | Out-Null
if(Test-Path (Join-Path $Work '.git')){
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P19.1'
  git -C $Work reset --hard FETCH_HEAD | Out-Host; Assert-LastExit 'git reset P19.1'
}else{
  git clone --no-checkout https://github.com/mrman123312/Leviathan.git $Work | Out-Host; Assert-LastExit 'git clone P19.1'
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P19.1'
  git -C $Work checkout --detach FETCH_HEAD | Out-Host; Assert-LastExit 'git checkout P19.1'
}

& $Py -c "import chess; print('python-chess',chess.__version__)"
if($LASTEXITCODE-ne 0){ & $Py -m pip install chess; Assert-LastExit 'install python-chess' }

$P09=Join-Path $Work 'src\stockfish.exe'
$bash='C:\msys64\usr\bin\bash.exe'
if(-not(Test-Path $bash)){throw 'MSYS2 bash missing at C:\msys64\usr\bin\bash.exe'}
$msysPath=$Work -replace '\\','/'
if($msysPath -match '^([A-Za-z]):/(.*)$'){$msysPath='/'+$matches[1].ToLower()+'/'+$matches[2]}
Write-Host '=== BUILD P19.1 FROM P09 STATIC-RACE CORE ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; cd '$msysPath/src' && make -j2 build ARCH=x86-64-avx2 COMP=mingw"
Assert-LastExit 'P19.1/P09 build'
if(-not(Test-Path $P09)){throw "P19.1 engine binary missing after build: $P09"}

$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){throw 'No stockfish-baseline.exe found under LeviathanHardwareResults.'}
$Stockfish=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'
Write-Host "Frozen Stockfish: $Stockfish" -ForegroundColor DarkGray

$Harness=Join-Path $Work 'tools\hybrid\run_p19_certified_survival_v2.py'
$Out=Join-Path $Work 'local_results\p19-certified-survival-v2'
Write-Host '=== 20 LOSS/MICRO SENTINELS, THEN FRESH 100-GAME ZERO-LOSS GATE ===' -ForegroundColor Magenta
Write-Host 'Normal: top-8 root census -> hostile post-move verification.' -ForegroundColor DarkGray
Write-Host 'Panic: all legal moves -> deep P09+Stockfish challenge -> top finalists.' -ForegroundColor DarkGray

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
  --root-nodes 24000 `
  --verify-nodes 16000 `
  --panic-root-nodes 120000 `
  --panic-verify-nodes 60000 `
  --finalist-nodes 240000 `
  --candidate-mpv 8 `
  --panic-finalists 6 `
  --safety-floor-cp -20 `
  --disagreement-cp 55 `
  --disagreement-weight 0.35 `
  --draw-lock-cp 120

$code=$LASTEXITCODE
if($code-eq 21){throw 'P19.1 REJECTED: a known loss/micro sentinel still produced an actual Leviathan loss.'}
if($code-eq 31){throw 'P19.1 REJECTED: a fresh Stockfish game produced an actual Leviathan loss.'}
if($code-eq 41){throw 'P19.1 NOT CERTIFIED: a sentinel reached a position where no move cleared the survival certificate.'}
if($code-eq 42){throw 'P19.1 NOT CERTIFIED: a fresh game reached a position where no move cleared the survival certificate.'}
Assert-LastExit 'P19.1 certified-survival gate'

Write-Host '=== P19.1 ZERO-LOSS GATE PASSED ===' -ForegroundColor Green
Write-Host 'This is empirical survival evidence over this adversarial domain, not a mathematical proof of chess.' -ForegroundColor Yellow
Write-Host 'Next stage: disagreement mining + sharper openings + deeper hostile critics.' -ForegroundColor Yellow
Write-Host "Results: $Out" -ForegroundColor Green
