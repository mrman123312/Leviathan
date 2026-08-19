$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }

$RepoCommit='3496bca7d3f3aef48a84b9adeddcd3474a9d9326'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$Work=Join-Path $Root 'p20-oracle-floor-work'
$Venv=Join-Path $Root 'p18-dml-venv'
$Py=Join-Path $Venv 'Scripts\python.exe'
if(-not(Test-Path $Py)){$Py=(Get-Command python -ErrorAction Stop).Source}

Write-Host '=== P20 ORACLE FLOOR / BASELINE DOMINANCE ===' -ForegroundColor Magenta
Write-Host 'P19 family is retired: evaluator confidence is not a safety proof.' -ForegroundColor Red
Write-Host 'New rule: Leviathan may improve on frozen Stockfish, but may not undercut its deeply verified defensive move.' -ForegroundColor Yellow
Write-Host 'No learned GPU authority. No master merge. Any actual loss aborts immediately.' -ForegroundColor Yellow
Write-Host "Pinned research source: $RepoCommit"

New-Item -ItemType Directory -Force -Path $Root | Out-Null
if(Test-Path (Join-Path $Work '.git')){
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P20'
  git -C $Work reset --hard FETCH_HEAD | Out-Host; Assert-LastExit 'git reset P20'
}else{
  git clone --no-checkout https://github.com/mrman123312/Leviathan.git $Work | Out-Host; Assert-LastExit 'git clone P20'
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P20'
  git -C $Work checkout --detach FETCH_HEAD | Out-Host; Assert-LastExit 'git checkout P20'
}

& $Py -c "import chess; print('python-chess',chess.__version__)"
if($LASTEXITCODE-ne 0){ & $Py -m pip install chess; Assert-LastExit 'install python-chess' }

$P09=Join-Path $Work 'src\stockfish.exe'
$bash='C:\msys64\usr\bin\bash.exe'
if(-not(Test-Path $bash)){throw 'MSYS2 bash missing at C:\msys64\usr\bin\bash.exe'}
$msysPath=$Work -replace '\\','/'
if($msysPath -match '^([A-Za-z]):/(.*)$'){$msysPath='/'+$matches[1].ToLower()+'/'+$matches[2]}
Write-Host '=== BUILD P20 FROM P09 STATIC-RACE CORE ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; cd '$msysPath/src' && make -j2 build ARCH=x86-64-avx2 COMP=mingw"
Assert-LastExit 'P20/P09 build'
if(-not(Test-Path $P09)){throw "P20 engine binary missing after build: $P09"}

$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){throw 'No stockfish-baseline.exe found under LeviathanHardwareResults.'}
$Stockfish=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'
Write-Host "Frozen Stockfish oracle/opponent: $Stockfish" -ForegroundColor DarkGray

$Harness=Join-Path $Work 'tools\hybrid\run_p20_oracle_floor.py'
$Out=Join-Path $Work 'local_results\p20-oracle-floor-v1'
Write-Host '=== 40 HISTORICAL/MICRO SENTINELS FIRST ===' -ForegroundColor Magenta
Write-Host '4 known failure positions x 10 repeats. First loss kills P20-v1.' -ForegroundColor Yellow
Write-Host 'Leviathan uses a much deeper defensive oracle than the 500ms opponent for this first invariant-establishment experiment.' -ForegroundColor Yellow

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
  --sentinel-repeats 10 `
  --floor-nodes 1000000 `
  --candidate-nodes 150000 `
  --verify-nodes 500000 `
  --reply-nodes 250000 `
  --continuation-nodes 500000 `
  --panic-nodes 2000000 `
  --candidate-mpv 8 `
  --reply-mpv 8 `
  --epsilon-cp 5

$code=$LASTEXITCODE
if($code-eq 21){throw 'P20-v1 REJECTED: a historical/micro sentinel still produced an actual Leviathan loss.'}
if($code-eq 31){throw 'P20-v1 REJECTED: a fresh Stockfish game produced an actual Leviathan loss.'}
Assert-LastExit 'P20 oracle-floor gate'

Write-Host '=== P20 ZERO-LOSS GATE PASSED ===' -ForegroundColor Green
Write-Host 'This establishes empirical baseline-dominance survival over the tested domain, not a mathematical solution of chess.' -ForegroundColor Yellow
Write-Host 'Next if it passes: add every new loss as a sentinel, adversarially mine sharp positions, then compress oracle cost.' -ForegroundColor Yellow
Write-Host "Results: $Out" -ForegroundColor Green
