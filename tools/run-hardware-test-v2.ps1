$ErrorActionPreference = 'Stop'

# Hotfix wrapper for the native Windows/MSYS2 Leviathan hardware tester.
# The original tester correctly clones/builds both engines, but MSYS2/UCRT
# Python does not expose the venv activation path the first revision assumed.
# Patch only that harness detail at runtime; engine source/mechanisms are untouched.

$BaseUrl = 'https://raw.githubusercontent.com/mrman123312/Leviathan/00346e2f9de98b863dab033c41814da0ea287fb4/tools/run-hardware-test.ps1'
Write-Host 'Loading Leviathan Windows hardware tester from GitHub...' -ForegroundColor Cyan
$Source = (Invoke-WebRequest -UseBasicParsing $BaseUrl).Content

$Old = @'
python -m venv "$VENV"
source "$VENV/Scripts/activate"
python -m pip install --disable-pip-version-check --upgrade pip >/dev/null
python -m pip install --disable-pip-version-check chess >/dev/null
'@

$New = @'
PYDEPS="$ROOT/pydeps"
mkdir -p "$PYDEPS"
python -m pip install --disable-pip-version-check --target "$PYDEPS" chess >/dev/null
export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PYVERIFY'
import chess, chess.engine
print('python-chess ready:', chess.__version__)
PYVERIFY
'@

if (-not $Source.Contains($Old)) {
    throw 'Tester hotfix could not find the expected venv block; refusing to run an unverified rewrite.'
}

$Patched = $Source.Replace($Old, $New)
Write-Host 'Applied MSYS2 Python hotfix. Starting zero-game hardware test...' -ForegroundColor Green
Invoke-Expression $Patched
