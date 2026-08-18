$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }

$RepoCommit='13f0268ba25719768f972ec031c32c1aed333f72'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$Work=Join-Path $Root 'p19-survival-work'
$Venv=Join-Path $Root 'p18-dml-venv'
$Py=Join-Path $Venv 'Scripts\python.exe'
if(-not(Test-Path $Py)){$Py=(Get-Command python -ErrorAction Stop).Source}

Write-Host '=== P19 SURVIVAL INVARIANT / ZERO-LOSS RESEARCH ===' -ForegroundColor Magenta
Write-Host 'Goal: eliminate losses, not merely improve average Elo.' -ForegroundColor Yellow
Write-Host 'Compute budget per Leviathan move: 4T+2T for 320ms, then 6T verifier for 180ms = 6T x 500ms.' -ForegroundColor Yellow
Write-Host 'No learned GPU authority. No ponder advantage. Any loss aborts immediately.' -ForegroundColor Yellow
Write-Host "Pinned research source: $RepoCommit"

New-Item -ItemType Directory -Force -Path $Root | Out-Null
if(Test-Path (Join-Path $Work '.git')){
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P19'
  git -C $Work reset --hard FETCH_HEAD | Out-Host; Assert-LastExit 'git reset P19'
}else{
  git clone --no-checkout https://github.com/mrman123312/Leviathan.git $Work | Out-Host; Assert-LastExit 'git clone P19'
  git -C $Work fetch --depth 1 origin $RepoCommit | Out-Host; Assert-LastExit 'git fetch P19'
  git -C $Work checkout --detach FETCH_HEAD | Out-Host; Assert-LastExit 'git checkout P19'
}

& $Py -c "import chess; print('python-chess',chess.__version__)"
if($LASTEXITCODE-ne 0){ & $Py -m pip install chess; Assert-LastExit 'install python-chess' }

$P09=Join-Path $Work 'src\stockfish.exe'
$bash='C:\msys64\usr\bin\bash.exe'
if(-not(Test-Path $bash)){throw 'MSYS2 bash missing at C:\msys64\usr\bin\bash.exe'}
$msysPath=$Work -replace '\\','/'
if($msysPath -match '^([A-Za-z]):/(.*)$'){$msysPath='/'+$matches[1].ToLower()+'/'+$matches[2]}
Write-Host '=== BUILD P19 FROM P09 STATIC-RACE CORE ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; cd '$msysPath/src' && make -j2 build ARCH=x86-64-avx2 COMP=mingw"
Assert-LastExit 'P19/P09 build'
if(-not(Test-Path $P09)){throw "P19 engine binary missing after build: $P09"}

$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){throw 'No stockfish-baseline.exe found under LeviathanHardwareResults.'}
$Stockfish=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'
Write-Host "Frozen Stockfish: $Stockfish" -ForegroundColor DarkGray

$Harness=Join-Path $Work 'tools\hybrid\run_p19_survival_match.py'
$Out=Join-Path $Work 'local_results\p19-survival-zero-loss'
Write-Host '=== RUN KNOWN-LOSS SENTINELS, THEN FRESH 100-GAME GATE ===' -ForegroundColor Magenta
& $Py $Harness --engine $P09 --opponent-engine $Stockfish --out-dir $Out --games 100 --threads 6 --hash 128 --movetime-ms 500 --broad-ms 320 --verify-ms 180 --max-plies 240 --opening-plies 10 --opening-nodes 1500 --seed 20260818 --sentinel-repeats 3 --missing-penalty 120 --disagreement-weight 0.35 --draw-lock-cp 80
$code=$LASTEXITCODE
if($code-eq 21){throw 'P19 rejected: a known-loss sentinel still beat the Survival Funnel.'}
if($code-eq 31){throw 'P19 rejected: a fresh Stockfish loss was found and saved as a new counterexample.'}
Assert-LastExit 'P19 survival zero-loss gate'

Write-Host '=== P19 ZERO-LOSS GATE PASSED ===' -ForegroundColor Green
Write-Host 'This is evidence over the tested domain, not a mathematical proof that chess cannot be lost.' -ForegroundColor Yellow
Write-Host "Results: $Out" -ForegroundColor Green
