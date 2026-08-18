$ErrorActionPreference = 'Stop'

# CPU-native wrapper around the existing Windows hardware test.
# The legacy harness hard-codes x86-64-avx2. This wrapper preserves the
# exact test logic while materializing the strongest architecture requested
# for BOTH Leviathan and Stockfish, so the comparison stays fair.
#
# Optional:
#   $env:LEVIATHAN_ARCH = 'x86-64-bmi2'  # or avxvnni/avx512/etc.
# Default:
#   native

$Arch = if ($env:LEVIATHAN_ARCH) { $env:LEVIATHAN_ARCH } else { 'native' }
$Source = Join-Path $PSScriptRoot 'run-hardware-test.ps1'
if (-not (Test-Path $Source)) { throw "Missing base harness: $Source" }

$Text = Get-Content -Raw -Path $Source
$Needle = 'ARCH=x86-64-avx2'
if (-not $Text.Contains($Needle)) {
    throw 'Base harness no longer contains the expected AVX2 build anchor; review before using this wrapper.'
}

$Text = $Text.Replace($Needle, "ARCH=$Arch")
$Text = $Text.Replace("echo 'AVX2 build failed; falling back to x86-64.'", "echo '$Arch build failed; falling back to x86-64.'")

$Temp = Join-Path $env:TEMP 'leviathan-hardware-test-native-materialized.ps1'
Set-Content -Encoding UTF8 -Path $Temp -Value $Text

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' LEVIATHAN CPU-NATIVE HARDWARE HARNESS' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host "Build architecture for BOTH engines: $Arch"
Write-Host "Materialized harness: $Temp"
Write-Host ''

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Temp
if ($LASTEXITCODE -ne 0) { throw "Native hardware harness failed with exit code $LASTEXITCODE" }
