$ErrorActionPreference = 'Stop'

# Architecture-aware hotfix wrapper for the native Windows/MSYS2 Leviathan tester.
# Defaults to ARCH=native so both Leviathan and the pinned Stockfish baseline are
# compiled for the actual CPU under test. Override with LEVIATHAN_ARCH when a
# portable or controlled ISA target is required (for example x86-64-avx2).

$BaseUrl = 'https://raw.githubusercontent.com/mrman123312/Leviathan/00346e2f9de98b863dab033c41814da0ea287fb4/tools/run-hardware-test.ps1'
$Arch = if ($env:LEVIATHAN_ARCH) { $env:LEVIATHAN_ARCH } else { 'native' }
if ($Arch -notmatch '^[A-Za-z0-9._+-]+$') {
    throw "Invalid LEVIATHAN_ARCH '$Arch'."
}

Write-Host 'Loading Leviathan Windows hardware tester from GitHub...' -ForegroundColor Cyan
Write-Host "Engine architecture target: $Arch" -ForegroundColor Cyan
$Source = (Invoke-WebRequest -UseBasicParsing $BaseUrl).Content

$OldPython = @'
python -m venv "$VENV"
source "$VENV/Scripts/activate"
python -m pip install --disable-pip-version-check --upgrade pip >/dev/null
python -m pip install --disable-pip-version-check chess >/dev/null
'@

$NewPython = @'
PYDEPS="$ROOT/pydeps"
mkdir -p "$PYDEPS"
python -m pip install --disable-pip-version-check --target "$PYDEPS" chess >/dev/null
export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PYVERIFY'
import chess, chess.engine
print('python-chess ready:', chess.__version__)
PYVERIFY
'@

if (-not $Source.Contains($OldPython)) {
    throw 'Tester hotfix could not find the expected venv block; refusing to run an unverified rewrite.'
}

$Patched = $Source.Replace($OldPython, $NewPython)
$Avx2Token = 'ARCH=x86-64-avx2'
if (-not $Patched.Contains($Avx2Token)) {
    throw 'Tester architecture hotfix could not find the expected AVX2 build token.'
}
$Patched = $Patched.Replace($Avx2Token, "ARCH=$Arch")

# Record the selected build target beside the hardware results. The underlying
# tester already records the CPU model, compiler, revisions, and binary hashes.
$BannerOld = "Write-Host 'Games:     0'"
$BannerNew = "Write-Host 'Games:     0'`nWrite-Host 'Build arch: $Arch'"
if ($Patched.Contains($BannerOld)) {
    $Patched = $Patched.Replace($BannerOld, $BannerNew)
}

Write-Host 'Applied MSYS2 Python + architecture hotfix. Starting zero-game hardware test...' -ForegroundColor Green
Invoke-Expression $Patched
