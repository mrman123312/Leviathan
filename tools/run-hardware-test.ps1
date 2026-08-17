$ErrorActionPreference = 'Stop'

# Leviathan local hardware tester.
# Downloads all source directly from GitHub. No existing local repository is required.
# Does not modify engine source, push, merge, tune, or play games.

$CandidateRepo = 'https://github.com/mrman123312/Leviathan.git'
$CandidateRef  = if ($env:LEVIATHAN_CANDIDATE_REF) { $env:LEVIATHAN_CANDIDATE_REF } else { 'leviathan/fundamentals-ultra-p01-qfrontier' }
$StockfishRepo = 'https://github.com/official-stockfish/Stockfish.git'
$StockfishSha  = '5062aee519a1ba262d472d8ab139851ced56573e'

$NodeBudget  = if ($env:LEVIATHAN_NODES) { [int]$env:LEVIATHAN_NODES } else { 100000 }
$TimeMs      = if ($env:LEVIATHAN_TIME_MS) { [int]$env:LEVIATHAN_TIME_MS } else { 100 }
$OracleNodes = if ($env:LEVIATHAN_ORACLE_NODES) { [int]$env:LEVIATHAN_ORACLE_NODES } else { 600000 }
$GradeNodes  = if ($env:LEVIATHAN_GRADE_NODES) { [int]$env:LEVIATHAN_GRADE_NODES } else { 300000 }

$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$WindowsResultRoot = Join-Path $HOME 'LeviathanHardwareResults'
$WindowsResults = Join-Path $WindowsResultRoot $Stamp
New-Item -ItemType Directory -Force -Path $WindowsResults | Out-Null

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' LEVIATHAN - GITHUB -> LOCAL HARDWARE TEST' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host "Candidate: $CandidateRef"
Write-Host "Stockfish: $StockfishSha"
Write-Host "Nodes:     $NodeBudget"
Write-Host "Time:      ${TimeMs}ms"
Write-Host "Results:   $WindowsResults"
Write-Host 'Games:     0'
Write-Host ''

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw @'
WSL is required for the Stockfish Linux build environment.
Open Administrator PowerShell and run:

    wsl --install

Restart Windows if requested, then run this tester again.
'@
}

Write-Host 'Windows GPU:' -ForegroundColor Yellow
if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
    nvidia-smi.exe
} else {
    Write-Warning 'nvidia-smi.exe is not on the Windows PATH.'
}

$WinResultUnix = $WindowsResults.Replace('\','/')
$WslResults = (& wsl.exe wslpath -a $WinResultUnix).Trim()
if (-not $WslResults) { throw 'Could not translate Windows result path into WSL path.' }

$Bash = @'
#!/usr/bin/env bash
set -euo pipefail

CANDIDATE_REPO="__CAND_REPO__"
CANDIDATE_REF="__CAND_REF__"
STOCKFISH_REPO="__SF_REPO__"
STOCKFISH_SHA="__SF_SHA__"
NODE_BUDGET=__NODES__
TIME_MS=__TIME_MS__
ORACLE_NODES=__ORACLE__
GRADE_NODES=__GRADE__
RESULTS="__RESULTS__"

ROOT="$HOME/.cache/leviathan-hardware-test"
CAND="$ROOT/candidate"
SF="$ROOT/stockfish"
VENV="$ROOT/venv"

mkdir -p "$ROOT" "$RESULTS"
rm -rf "$CAND" "$SF"

if ! command -v git >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    echo 'Installing Linux build prerequisites...'
    sudo apt-get update
    sudo apt-get install -y build-essential git python3 python3-pip python3-venv
fi

echo
echo '=== HARDWARE ==='
(lscpu || true) | tee "$RESULTS/cpu.txt"
(free -h || true) | tee "$RESULTS/memory.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee "$RESULTS/nvidia-smi-wsl.txt"
else
    echo 'nvidia-smi unavailable inside WSL' | tee "$RESULTS/nvidia-smi-wsl.txt"
fi

echo
echo '=== CLONE CANDIDATE DIRECTLY FROM GITHUB ==='
git clone --filter=blob:none --no-checkout "$CANDIDATE_REPO" "$CAND"
git -C "$CAND" fetch origin "$CANDIDATE_REF"
git -C "$CAND" checkout --detach FETCH_HEAD
CAND_SHA=$(git -C "$CAND" rev-parse HEAD)
echo "candidate_sha=$CAND_SHA" | tee "$RESULTS/revisions.txt"

echo
echo '=== CLONE PINNED STOCKFISH DIRECTLY FROM GITHUB ==='
git clone --filter=blob:none --no-checkout "$STOCKFISH_REPO" "$SF"
git -C "$SF" fetch origin "$STOCKFISH_SHA"
git -C "$SF" checkout --detach FETCH_HEAD
SF_ACTUAL_SHA=$(git -C "$SF" rev-parse HEAD)
echo "stockfish_sha=$SF_ACTUAL_SHA" | tee -a "$RESULTS/revisions.txt"

build_engine() {
    local dir="$1"
    local out="$2"
    cd "$dir/src"
    make net || true
    make clean
    if ! make -j"$(nproc)" build ARCH=x86-64-avx2; then
        echo 'AVX2 build failed; falling back to x86-64.'
        make clean
        make -j"$(nproc)" build ARCH=x86-64
    fi
    test -x stockfish
    cp stockfish "$RESULTS/$out"
}

echo
echo '=== BUILD STOCKFISH ==='
build_engine "$SF" stockfish-baseline

echo
echo '=== BUILD LEVIATHAN ==='
build_engine "$CAND" leviathan-candidate

BASE_EXE="$RESULTS/stockfish-baseline"
CAND_EXE="$RESULTS/leviathan-candidate"
sha256sum "$BASE_EXE" "$CAND_EXE" | tee "$RESULTS/binaries.sha256"

rm -rf "$VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install chess >/dev/null

python - <<'PY' | tee "$RESULTS/python-gpu-check.txt"
try:
    import torch
    print('torch_version =', torch.__version__)
    print('cuda_available =', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('gpu =', torch.cuda.get_device_name(0))
        print('cuda =', torch.version.cuda)
except Exception as exc:
    print('PyTorch is not installed in this clean tester venv; GPU inference test skipped:', repr(exc))
PY

cat > "$ROOT/test_positions.py" <<'PY'
import argparse
import json
import statistics
import time
from pathlib import Path
import chess
import chess.engine

FENS = [
    chess.STARTING_FEN,
    'r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1',
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
    'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
    'rnbq1k1r/pp1Pbppp/2p2n2/8/2B5/8/PPP1NPPP/RNBQK2R w KQ - 1 8',
    'r4rk1/1pp1qppp/p1np1n2/8/2B1P3/2N1Q3/PPP2PPP/R4RK1 w - - 0 10',
    'r1bq1rk1/pp2bppp/2n1pn2/2pp4/8/1P1PPN2/PBPN1PPP/R2Q1RK1 w - - 2 10',
    'r2q1rk1/pp2bppp/2npbn2/2p5/4P3/1NN1B3/PPPQ1PPP/2KR1B1R w - - 4 11',
    '2rq1rk1/pp2bppp/3p1n2/2pPp3/4P3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 13',
    '8/5pk1/5np1/3p4/3P4/5P2/5KPP/8 w - - 0 40',
    '8/8/5pk1/5np1/3P4/5P2/4K1PP/8 w - - 0 42',
]

def cfg(e):
    o = {}
    if 'Threads' in e.options: o['Threads'] = 1
    if 'Hash' in e.options: o['Hash'] = 64
    if 'Leviathan Fundamentals' in e.options: o['Leviathan Fundamentals'] = True
    if 'Leviathan Fundamentals Authority' in e.options: o['Leviathan Fundamentals Authority'] = 1
    if 'Leviathan Quiet Overdrive' in e.options: o['Leviathan Quiet Overdrive'] = 0
    for n in ['Leviathan Risk','Leviathan Policy','Leviathan MetaSearch','Leviathan Specialist','Leviathan Atlas','Leviathan Search DSL']:
        if n in e.options: o[n] = False
    if o: e.configure(o)

def score(info, color):
    s = info.get('score')
    if s is None: return None
    return s.pov(color).score(mate_score=100000)

def choose(e, b, limit, token):
    t0 = time.perf_counter()
    r = e.play(b, limit, info=chess.engine.INFO_ALL, game=token)
    return {'move': r.move, 'nodes': r.info.get('nodes'), 'depth': r.info.get('depth'), 'seldepth': r.info.get('seldepth'), 'elapsed_ms': (time.perf_counter()-t0)*1000}

def grade(e, b, m, nodes, token):
    if m is None: return None
    root = b.turn
    x = b.copy(); x.push(m)
    i = e.analyse(x, chess.engine.Limit(nodes=nodes), game=token)
    return score(i, root)

def mean(rows, key):
    v=[r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(v) if v else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True); p.add_argument('--baseline',required=True); p.add_argument('--output',required=True)
    p.add_argument('--nodes',type=int,required=True); p.add_argument('--time-ms',type=int,required=True); p.add_argument('--oracle-nodes',type=int,required=True); p.add_argument('--grade-nodes',type=int,required=True)
    a=p.parse_args()
    c=chess.engine.SimpleEngine.popen_uci(a.candidate); b=chess.engine.SimpleEngine.popen_uci(a.baseline); o=chess.engine.SimpleEngine.popen_uci(a.baseline)
    for e in (c,b,o): cfg(e)
    rows=[]
    try:
        for k,fen in enumerate(FENS):
            pos=chess.Board(fen); color=pos.turn
            print(f'[{k+1}/{len(FENS)}] {fen}',flush=True)
            oi=o.analyse(pos,chess.engine.Limit(nodes=a.oracle_nodes),game=('oracle',k)); os=score(oi,color); om=oi.get('pv',[None])[0] if oi.get('pv') else None
            bn=choose(b,pos,chess.engine.Limit(nodes=a.nodes),('bn',k)); cn=choose(c,pos,chess.engine.Limit(nodes=a.nodes),('cn',k))
            bg=grade(o,pos,bn['move'],a.grade_nodes,('bg',k)); cg=grade(o,pos,cn['move'],a.grade_nodes,('cg',k))
            bt=choose(b,pos,chess.engine.Limit(time=a.time_ms/1000),('bt',k)); ct=choose(c,pos,chess.engine.Limit(time=a.time_ms/1000),('ct',k))
            btg=grade(o,pos,bt['move'],a.grade_nodes,('btg',k)); ctg=grade(o,pos,ct['move'],a.grade_nodes,('ctg',k))
            regret=lambda g: None if g is None or os is None else max(0,os-g)
            rows.append({'fen':fen,'oracle_move':str(om),'equal_node_baseline_move':str(bn['move']),'equal_node_candidate_move':str(cn['move']),'equal_node_baseline_regret':regret(bg),'equal_node_candidate_regret':regret(cg),'equal_node_baseline_ms':bn['elapsed_ms'],'equal_node_candidate_ms':cn['elapsed_ms'],'fixed_time_baseline_move':str(bt['move']),'fixed_time_candidate_move':str(ct['move']),'fixed_time_baseline_regret':regret(btg),'fixed_time_candidate_regret':regret(ctg),'fixed_time_baseline_nodes':bt['nodes'],'fixed_time_candidate_nodes':ct['nodes'],'fixed_time_baseline_depth':bt['depth'],'fixed_time_candidate_depth':ct['depth']})
    finally:
        c.quit(); b.quit(); o.quit()
    summary={'positions':len(rows),'equal_node':{'baseline_mean_regret_cp':mean(rows,'equal_node_baseline_regret'),'candidate_mean_regret_cp':mean(rows,'equal_node_candidate_regret'),'baseline_mean_ms':mean(rows,'equal_node_baseline_ms'),'candidate_mean_ms':mean(rows,'equal_node_candidate_ms')},'fixed_time':{'baseline_mean_regret_cp':mean(rows,'fixed_time_baseline_regret'),'candidate_mean_regret_cp':mean(rows,'fixed_time_candidate_regret'),'baseline_mean_nodes':mean(rows,'fixed_time_baseline_nodes'),'candidate_mean_nodes':mean(rows,'fixed_time_candidate_nodes'),'baseline_mean_depth':mean(rows,'fixed_time_baseline_depth'),'candidate_mean_depth':mean(rows,'fixed_time_candidate_depth')}}
    summary['equal_node']['regret_improvement_cp']=summary['equal_node']['baseline_mean_regret_cp']-summary['equal_node']['candidate_mean_regret_cp']
    summary['fixed_time']['regret_improvement_cp']=summary['fixed_time']['baseline_mean_regret_cp']-summary['fixed_time']['candidate_mean_regret_cp']
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'rows.json').write_text(json.dumps(rows,indent=2)); (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
PY

echo
echo '=== ZERO-GAME POSITION TEST ==='
python "$ROOT/test_positions.py" \
    --candidate "$CAND_EXE" \
    --baseline "$BASE_EXE" \
    --output "$RESULTS" \
    --nodes "$NODE_BUDGET" \
    --time-ms "$TIME_MS" \
    --oracle-nodes "$ORACLE_NODES" \
    --grade-nodes "$GRADE_NODES" \
    | tee "$RESULTS/run.log"

echo
echo '=== COMPLETE ==='
echo "Results: $RESULTS"
cat "$RESULTS/summary.json"
echo
echo 'NO GAMES PLAYED. NO SOURCE MODIFIED. NO PUSH. NO MERGE.'
'@

$Bash = $Bash.Replace('__CAND_REPO__', $CandidateRepo)
$Bash = $Bash.Replace('__CAND_REF__', $CandidateRef)
$Bash = $Bash.Replace('__SF_REPO__', $StockfishRepo)
$Bash = $Bash.Replace('__SF_SHA__', $StockfishSha)
$Bash = $Bash.Replace('__NODES__', "$NodeBudget")
$Bash = $Bash.Replace('__TIME_MS__', "$TimeMs")
$Bash = $Bash.Replace('__ORACLE__', "$OracleNodes")
$Bash = $Bash.Replace('__GRADE__', "$GradeNodes")
$Bash = $Bash.Replace('__RESULTS__', $WslResults)
$Bash = $Bash -replace "`r`n", "`n"

$Temp = Join-Path $env:TEMP 'leviathan-hardware-test.sh'
[System.IO.File]::WriteAllText($Temp, $Bash, [System.Text.UTF8Encoding]::new($false))
$TempUnix = $Temp.Replace('\','/')
$WslTemp = (& wsl.exe wslpath -a $TempUnix).Trim()
if (-not $WslTemp) { throw 'Could not translate temporary script path into WSL path.' }

Write-Host ''
Write-Host 'Starting WSL build/test. Source will be cloned directly from GitHub...' -ForegroundColor Green
& wsl.exe bash $WslTemp
if ($LASTEXITCODE -ne 0) { throw "Leviathan hardware test failed with exit code $LASTEXITCODE" }

Write-Host ''
Write-Host 'DONE.' -ForegroundColor Green
Write-Host "Results are in: $WindowsResults"
Write-Host "Open: $(Join-Path $WindowsResults 'summary.json')"
