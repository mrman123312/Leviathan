$ErrorActionPreference='Stop'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){ throw "No stockfish-baseline.exe found under $Root. Run the hardware build first." }
$base=Join-Path $baselineDir.FullName 'stockfish-baseline.exe'

$work=Join-Path $Root 'p18.2-one-shot-work'
if(Test-Path $work){ Remove-Item -Recurse -Force $work }
Write-Host '=== CLONE CURRENT P18.2 / P09 SOURCE ===' -ForegroundColor Cyan
git clone --depth 1 --branch agent/p18-hybrid-cpu-gpu-multiponder https://github.com/mrman123312/Leviathan.git $work | Out-Host
if($LASTEXITCODE-ne 0){ throw 'Git clone failed' }

$bash='C:\msys64\usr\bin\bash.exe'
$compiler='C:\msys64\ucrt64\bin\x86_64-w64-mingw32-c++.exe'
if(-not (Test-Path $bash)){ throw 'MSYS2 bash not found at C:\msys64\usr\bin\bash.exe' }
if(-not (Test-Path $compiler)){ throw 'MSYS2 UCRT64 compiler not found at C:\msys64\ucrt64\bin\x86_64-w64-mingw32-c++.exe. Install the UCRT64 GCC toolchain first.' }
$msysPath=$work -replace '\\','/'
if($msysPath -match '^([A-Za-z]):/(.*)$'){ $msysPath='/' + $matches[1].ToLower() + '/' + $matches[2] }
Write-Host '=== VERIFY UCRT64 TOOLCHAIN ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; command -v x86_64-w64-mingw32-c++; x86_64-w64-mingw32-c++ --version | head -n 1"
if($LASTEXITCODE-ne 0){ throw 'UCRT64 compiler exists but is not executable inside MSYS2.' }
Write-Host '=== BUILD CURRENT P09 STATIC-RACE CORE ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; cd '$msysPath/src' && make -j2 build ARCH=x86-64-avx2 COMP=mingw"
if($LASTEXITCODE-ne 0){ throw 'P09 build failed' }
$cand=Join-Path $work 'src\stockfish.exe'
if(-not (Test-Path $cand)){ throw "Built engine not found: $cand" }

Write-Host "P09 engine: $cand" -ForegroundColor Green
Write-Host "Stockfish baseline: $base" -ForegroundColor Green
Write-Host '=== START P18.2 TRAIN + PROSPECTIVE HOLDOUT + WARM-SEARCH GATE ===' -ForegroundColor Cyan
$runner=Join-Path $work 'tools\hybrid\run-hybrid-one-shot.ps1'
& $runner -Engine $cand -OpponentEngine $base -Threads 8 -Hash 128 -Games 80 -OutDir 'local_results/hybrid/p18-one-shot'
if($LASTEXITCODE-ne 0){ throw 'P18.2 one-shot campaign failed a gate. Copy the console output back into ChatGPT.' }
Write-Host "`nP18.2 ONE-SHOT COMPLETE" -ForegroundColor Green
Write-Host "Work tree: $work" -ForegroundColor Green
Write-Host "Results: $(Join-Path $work 'local_results\hybrid\p18-one-shot')" -ForegroundColor Green