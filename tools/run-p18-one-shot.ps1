$ErrorActionPreference='Stop'
function Assert-LastExit([string]$Stage){ if($LASTEXITCODE-ne 0){ throw "$Stage failed with exit code $LASTEXITCODE" } }
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$baselineDir=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe') } | Select-Object -First 1
if(-not $baselineDir){ throw 'No stockfish-baseline.exe found under LeviathanHardwareResults.' }
$base=Join-Path $baselineDir.FullName 'stockfish-baseline.exe';$work=Join-Path $Root 'p18.2-one-shot-work'
if(Test-Path (Join-Path $work '.git')){
  Write-Host '=== UPDATE CURRENT P18.3 SOURCE (PRESERVE LOCAL RESULTS) ===' -ForegroundColor Cyan
  git -C $work fetch --depth 1 origin agent/p18-hybrid-cpu-gpu-multiponder | Out-Host;Assert-LastExit 'git fetch'
  git -C $work reset --hard FETCH_HEAD | Out-Host;Assert-LastExit 'git reset'
}else{
  if(Test-Path $work){Remove-Item -Recurse -Force $work}
  Write-Host '=== CLONE CURRENT P18.3 / P09 SOURCE ===' -ForegroundColor Cyan
  git clone --depth 1 --branch agent/p18-hybrid-cpu-gpu-multiponder https://github.com/mrman123312/Leviathan.git $work | Out-Host;Assert-LastExit 'git clone'
}
$bash='C:\msys64\usr\bin\bash.exe';$compiler='C:\msys64\ucrt64\bin\x86_64-w64-mingw32-c++.exe'
if(-not (Test-Path $bash)){throw 'MSYS2 bash missing'};if(-not (Test-Path $compiler)){throw 'MSYS2 UCRT64 compiler missing'}
$msysPath=$work -replace '\\','/';if($msysPath -match '^([A-Za-z]):/(.*)$'){$msysPath='/'+$matches[1].ToLower()+'/'+$matches[2]}
Write-Host '=== VERIFY UCRT64 TOOLCHAIN ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; command -v x86_64-w64-mingw32-c++; x86_64-w64-mingw32-c++ --version | head -n 1";Assert-LastExit 'UCRT64 verification'
Write-Host '=== BUILD CURRENT P09 STATIC-RACE CORE ===' -ForegroundColor Cyan
& $bash -lc "export PATH=/ucrt64/bin:/usr/bin:`$PATH; cd '$msysPath/src' && make -j2 build ARCH=x86-64-avx2 COMP=mingw";Assert-LastExit 'P09 build'
$cand=Join-Path $work 'src\stockfish.exe';if(-not(Test-Path $cand)){throw "Built engine missing: $cand"}

Write-Host '=== PREPARE LIGHTWEIGHT DIRECTML GPU ENVIRONMENT ===' -ForegroundColor Cyan
$hostPython=(Get-Command python -ErrorAction Stop).Source;$venv=Join-Path $Root 'p18-dml-venv';$venvPy=Join-Path $venv 'Scripts\python.exe'
if(-not(Test-Path $venvPy)){ & $hostPython -m venv $venv;Assert-LastExit 'DML virtual environment creation' }
$dmlReady=$false
& $venvPy -c "import torch,torch_directml; d=torch_directml.device(); x=torch.tensor([1.,2.]).to(d); print('existing DirectML',torch.__version__,str(d),float((x*x).sum().cpu()));" 2>$null
if($LASTEXITCODE-eq 0){$dmlReady=$true}
if(-not $dmlReady){
  Write-Host 'Installing torch-directml (small Windows GPU bridge; no 3.5 GB CUDA wheel)...' -ForegroundColor Yellow
  & $venvPy -m pip install --disable-pip-version-check --no-cache-dir torch-directml
  Assert-LastExit 'torch-directml install'
}
& $venvPy -c "import json,torch,torch_directml; d=torch_directml.device(); x=torch.tensor([[1.,2.],[3.,4.]]).to(d); y=(x@x).cpu(); print(json.dumps({'python':__import__('sys').version.split()[0],'torch':torch.__version__,'accelerator':'DirectML','device':str(d),'probe':y.tolist()},indent=2))"
Assert-LastExit 'DirectML verification'

Write-Host "P09 engine: $cand" -ForegroundColor Green;Write-Host "Stockfish baseline: $base" -ForegroundColor Green;Write-Host "P18 GPU Python: $venvPy" -ForegroundColor Green
Write-Host '=== START/RESUME P18.3 TRAIN + WHOLE-GAME PROSPECTIVE HOLDOUT + WARM-SEARCH GATE ===' -ForegroundColor Cyan
$runner=Join-Path $work 'tools\hybrid\run-hybrid-one-shot.ps1'
& $runner -Engine $cand -OpponentEngine $base -Threads 8 -Hash 128 -Games 80 -Python $venvPy -Device dml -OutDir 'local_results/hybrid/p18-one-shot'
if($LASTEXITCODE-ne 0){throw 'P18.3 one-shot campaign failed a gate. Copy the console output back into ChatGPT.'}
Write-Host "`nP18.3 ONE-SHOT COMPLETE" -ForegroundColor Green;Write-Host "Work tree: $work" -ForegroundColor Green;Write-Host "Results: $(Join-Path $work 'local_results\hybrid\p18-one-shot')" -ForegroundColor Green