$ErrorActionPreference='Stop'

# Stable match-first launcher.
# Reuses the existing match-first orchestration but disables dynamic ponder-pool annealing
# for the 100-game benchmark. The learned GPU advisor still ranks the candidate pool and
# controls the initial thread allocation; only the unsafe 150 ms stop/reconfigure loop is removed.
# Because anneal_seconds is part of the harness identity, this produces a fresh run_id and
# cannot reuse games from the earlier annealing-contaminated run.

$Base='https://raw.githubusercontent.com/mrman123312/Leviathan/f1fc039ea30e3f81f45b35c9ae820134f17e1393/tools/run-p18-vs-stockfish-100.ps1'
$src=irm $Base
$old='--anneal-seconds 0.15 --min-final-scouts 2'
$new='--anneal-seconds 0 --min-final-scouts 2'
if(-not $src.Contains($old)){ throw 'Pinned match-first launcher no longer contains the expected annealing argument; refusing an ambiguous rewrite.' }
$src=$src.Replace($old,$new)
$src=$src.Replace("Write-Host 'Both variants keep P18 multi-ponder; the ablation disables the learned GPU/model advisor only.'", "Write-Host 'Both variants keep the same fixed P18 multi-ponder portfolio; dynamic annealing is disabled for benchmark stability. The ablation disables the learned GPU/model advisor only.'")
Write-Host '=== STABILITY OVERRIDE: FIXED PONDER PORTFOLIO / NO DYNAMIC ANNEAL RECONFIGURATION ===' -ForegroundColor Green
Write-Host 'GPU advisor still ranks candidates and allocates initial scout threads. GPU-off decisive replays use the identical fixed-portfolio architecture.' -ForegroundColor Green
Invoke-Expression $src
