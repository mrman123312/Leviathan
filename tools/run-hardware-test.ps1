$ErrorActionPreference = 'Stop'

# Leviathan local hardware tester.
# Pulls all source directly from GitHub.
# Native Windows/PowerShell launcher; WSL is NOT required.
# Installs MSYS2/MinGW through winget if needed.
# Does not modify engine source, tune mechanisms, push, merge, or play games.

$CandidateRepo = 'https://github.com/mrman123312/Leviathan.git'
$CandidateRef  = if ($env:LEVIATHAN_CANDIDATE_REF) { $env:LEVIATHAN_CANDIDATE_REF } else { 'leviathan/fundamentals-ultra-p01-qfrontier' }
$StockfishRepo = 'https://github.com/official-stockfish/Stockfish.git'
$StockfishSha  = '5062aee519a1ba262d472d8ab139851ced56573e'

$NodeBudget  = if ($env:LEVIATHAN_NODES) { [int]$env:LEVIATHAN_NODES } else { 100000 }
$TimeMs      = if ($env:LEVIATHAN_TIME_MS) { [int]$env:LEVIATHAN_TIME_MS } else { 100 }
$OracleNodes = if ($env:LEVIATHAN_ORACLE_NODES) { [int]$env:LEVIATHAN_ORACLE_NODES } else { 600000 }
$GradeNodes  = if ($env:LEVIATHAN_GRADE_NODES) { [int]$env:LEVIATHAN_GRADE_NODES } else { 300000 }

$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$ResultRoot = Join-Path $HOME 'LeviathanHardwareResults'
$Results = Join-Path $ResultRoot $Stamp
New-Item -ItemType Directory -Force -Path $Results | Out-Null

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' LEVIATHAN - GITHUB -> WINDOWS HARDWARE TEST' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host "Candidate: $CandidateRef"
Write-Host "Stockfish: $StockfishSha"
Write-Host "Nodes:     $NodeBudget"
Write-Host "Time:      ${TimeMs}ms"
Write-Host "Results:   $Results"
Write-Host 'Games:     0'
Write-Host 'WSL:       not required'
Write-Host ''

try {
    Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | Format-List | Out-File -Encoding utf8 (Join-Path $Results 'cpu-windows.txt')
    Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,Manufacturer,Model | Format-List | Out-File -Append -Encoding utf8 (Join-Path $Results 'cpu-windows.txt')
} catch { $_ | Out-File -Encoding utf8 (Join-Path $Results 'cpu-windows.txt') }

Write-Host 'Windows GPU:' -ForegroundColor Yellow
if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
    (& nvidia-smi.exe) | Tee-Object -FilePath (Join-Path $Results 'nvidia-smi.txt')
} else {
    'nvidia-smi.exe not found on PATH' | Tee-Object -FilePath (Join-Path $Results 'nvidia-smi.txt')
}

$MsysRoot = 'C:\msys64'
$Bash = Join-Path $MsysRoot 'usr\bin\bash.exe'

if (-not (Test-Path $Bash)) {
    Write-Host ''
    Write-Host 'MSYS2 not found. Installing it automatically with winget...' -ForegroundColor Yellow
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw 'winget is unavailable. Install/update App Installer from Microsoft Store, then rerun this one-line tester.'
    }
    & winget.exe install --id MSYS2.MSYS2 -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "MSYS2 installation failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path $Bash)) { throw "MSYS2 installation completed but $Bash was not found. Open MSYS2 once, then rerun." }
}

Write-Host ''
Write-Host 'Preparing MSYS2 compiler/Python environment...' -ForegroundColor Yellow
$PackageCommand = @'
set -e
export PATH=/ucrt64/bin:/usr/bin:$PATH
pacman -Sy --noconfirm
pacman -S --noconfirm --needed git make curl ca-certificates mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-python mingw-w64-ucrt-x86_64-python-pip
'@
& $Bash -lc $PackageCommand
if ($LASTEXITCODE -ne 0) { throw "MSYS2 package setup failed with exit code $LASTEXITCODE" }

$EscapedResults = $Results.Replace("'", "'\''")
$ResultUnix = (& $Bash -lc "cygpath -u '$EscapedResults'").Trim()
if (-not $ResultUnix) { throw 'Could not convert the result directory to an MSYS2 path.' }

$RunnerWin = Join-Path $env:TEMP 'leviathan-native-runner.sh'

$Runner = @'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/ucrt64/bin:/usr/bin:$PATH

CANDIDATE_REPO="__CAND_REPO__"
CANDIDATE_REF="__CAND_REF__"
STOCKFISH_REPO="__SF_REPO__"
STOCKFISH_SHA="__SF_SHA__"
NODE_BUDGET=__NODES__
TIME_MS=__TIME_MS__
ORACLE_NODES=__ORACLE__
GRADE_NODES=__GRADE__
RESULTS="__RESULTS__"

ROOT="/tmp/leviathan-hardware-test-$$"
CAND="$ROOT/candidate"
SF="$ROOT/stockfish"
VENV="$ROOT/venv"
mkdir -p "$ROOT" "$RESULTS"
cleanup(){ rm -rf "$ROOT" || true; }
trap cleanup EXIT

echo
echo '=== TOOLCHAIN ==='
uname -a | tee "$RESULTS/msys-uname.txt"
g++ --version | head -n 1 | tee "$RESULTS/compiler.txt"
python --version | tee "$RESULTS/python-version.txt"

echo
echo '=== CLONE LEVIATHAN DIRECTLY FROM GITHUB ==='
git clone --filter=blob:none --no-checkout "$CANDIDATE_REPO" "$CAND"
git -C "$CAND" fetch origin "$CANDIDATE_REF"
git -C "$CAND" checkout --detach FETCH_HEAD
CAND_SHA=$(git -C "$CAND" rev-parse HEAD)
printf 'candidate_ref=%s\ncandidate_sha=%s\n' "$CANDIDATE_REF" "$CAND_SHA" | tee "$RESULTS/revisions.txt"

echo
echo '=== CLONE PINNED STOCKFISH DIRECTLY FROM GITHUB ==='
git clone --filter=blob:none --no-checkout "$STOCKFISH_REPO" "$SF"
git -C "$SF" fetch origin "$STOCKFISH_SHA"
git -C "$SF" checkout --detach FETCH_HEAD
SF_ACTUAL_SHA=$(git -C "$SF" rev-parse HEAD)
printf 'stockfish_sha=%s\n' "$SF_ACTUAL_SHA" | tee -a "$RESULTS/revisions.txt"

JOBS=${NUMBER_OF_PROCESSORS:-4}

build_engine(){
  local dir="$1"; local out="$2"; cd "$dir/src"
  make net || true
  make clean || true
  if make -j"$JOBS" build ARCH=x86-64-avx2 COMP=mingw; then :
  elif make -j"$JOBS" build ARCH=x86-64-avx2; then :
  else
    echo 'AVX2 build failed; falling back to x86-64.'
    make clean || true
    if ! make -j"$JOBS" build ARCH=x86-64 COMP=mingw; then make -j"$JOBS" build ARCH=x86-64; fi
  fi
  local exe=''
  if [[ -x stockfish.exe ]]; then exe='stockfish.exe'; elif [[ -x stockfish ]]; then exe='stockfish'; else exe=$(find . -maxdepth 1 -type f \( -name 'stockfish*.exe' -o -name 'stockfish' \) | head -n 1 || true); fi
  [[ -n "$exe" ]] || { echo "No built engine found in $dir/src"; exit 2; }
  cp "$exe" "$RESULTS/$out.exe"
}

echo
echo '=== BUILD STOCKFISH ==='
build_engine "$SF" stockfish-baseline

echo
echo '=== BUILD LEVIATHAN ==='
build_engine "$CAND" leviathan-candidate

BASE_EXE="$RESULTS/stockfish-baseline.exe"
CAND_EXE="$RESULTS/leviathan-candidate.exe"
sha256sum "$BASE_EXE" "$CAND_EXE" | tee "$RESULTS/binaries.sha256"

python -m venv "$VENV"
source "$VENV/Scripts/activate"
python -m pip install --disable-pip-version-check --upgrade pip >/dev/null
python -m pip install --disable-pip-version-check chess >/dev/null

cat > "$ROOT/test_positions.py" <<'PY'
import argparse,json,statistics,time
from pathlib import Path
import chess,chess.engine
FENS=[chess.STARTING_FEN,'r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1','8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1','r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1','rnbq1k1r/pp1Pbppp/2p2n2/8/2B5/8/PPP1NPPP/RNBQK2R w KQ - 1 8','r4rk1/1pp1qppp/p1np1n2/8/2B1P3/2N1Q3/PPP2PPP/R4RK1 w - - 0 10','r1bq1rk1/pp2bppp/2n1pn2/2pp4/8/1P1PPN2/PBPN1PPP/R2Q1RK1 w - - 2 10','r2q1rk1/pp2bppp/2npbn2/2p5/4P3/1NN1B3/PPPQ1PPP/2KR1B1R w - - 4 11','2rq1rk1/pp2bppp/3p1n2/2pPp3/4P3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 13','8/5pk1/5np1/3p4/3P4/5P2/5KPP/8 w - - 0 40','8/8/5pk1/5np1/3P4/5P2/4K1PP/8 w - - 0 42']
def cfg(e):
 o={}
 if 'Threads' in e.options:o['Threads']=1
 if 'Hash' in e.options:o['Hash']=64
 if 'Leviathan Fundamentals' in e.options:o['Leviathan Fundamentals']=True
 if 'Leviathan Fundamentals Authority' in e.options:o['Leviathan Fundamentals Authority']=1
 if 'Leviathan Quiet Overdrive' in e.options:o['Leviathan Quiet Overdrive']=0
 for n in ['Leviathan Risk','Leviathan Policy','Leviathan MetaSearch','Leviathan Specialist','Leviathan Atlas','Leviathan Search DSL']:
  if n in e.options:o[n]=False
 if o:e.configure(o)
def cp(i,c):
 s=i.get('score');return None if s is None else s.pov(c).score(mate_score=100000)
def choose(e,b,l,tok):
 t=time.perf_counter();r=e.play(b,l,info=chess.engine.INFO_ALL,game=tok);return {'move':r.move,'nodes':r.info.get('nodes'),'depth':r.info.get('depth'),'elapsed_ms':(time.perf_counter()-t)*1000}
def grade(e,b,m,n,tok):
 if m is None:return None
 root=b.turn;x=b.copy();x.push(m);return cp(e.analyse(x,chess.engine.Limit(nodes=n),game=tok),root)
def mean(rs,k):
 v=[r[k] for r in rs if r.get(k) is not None];return statistics.mean(v) if v else None
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--nodes',type=int,required=True);p.add_argument('--time-ms',type=int,required=True);p.add_argument('--oracle-nodes',type=int,required=True);p.add_argument('--grade-nodes',type=int,required=True);a=p.parse_args()
 c=chess.engine.SimpleEngine.popen_uci(a.candidate);b=chess.engine.SimpleEngine.popen_uci(a.baseline);o=chess.engine.SimpleEngine.popen_uci(a.baseline)
 for e in(c,b,o):cfg(e)
 rows=[]
 try:
  for k,fen in enumerate(FENS):
   pos=chess.Board(fen);col=pos.turn;print(f'[{k+1}/{len(FENS)}] {fen}',flush=True);oi=o.analyse(pos,chess.engine.Limit(nodes=a.oracle_nodes),game=('oracle',k));os=cp(oi,col);om=oi.get('pv',[None])[0] if oi.get('pv') else None
   bn=choose(b,pos,chess.engine.Limit(nodes=a.nodes),('bn',k));cn=choose(c,pos,chess.engine.Limit(nodes=a.nodes),('cn',k));bg=grade(o,pos,bn['move'],a.grade_nodes,('bg',k));cg=grade(o,pos,cn['move'],a.grade_nodes,('cg',k));bt=choose(b,pos,chess.engine.Limit(time=a.time_ms/1000),('bt',k));ct=choose(c,pos,chess.engine.Limit(time=a.time_ms/1000),('ct',k));btg=grade(o,pos,bt['move'],a.grade_nodes,('btg',k));ctg=grade(o,pos,ct['move'],a.grade_nodes,('ctg',k));reg=lambda g:None if g is None or os is None else max(0,os-g)
   rows.append({'fen':fen,'oracle_move':str(om),'equal_node_baseline_move':str(bn['move']),'equal_node_candidate_move':str(cn['move']),'equal_node_baseline_regret':reg(bg),'equal_node_candidate_regret':reg(cg),'equal_node_baseline_ms':bn['elapsed_ms'],'equal_node_candidate_ms':cn['elapsed_ms'],'fixed_time_baseline_move':str(bt['move']),'fixed_time_candidate_move':str(ct['move']),'fixed_time_baseline_regret':reg(btg),'fixed_time_candidate_regret':reg(ctg),'fixed_time_baseline_nodes':bt['nodes'],'fixed_time_candidate_nodes':ct['nodes'],'fixed_time_baseline_depth':bt['depth'],'fixed_time_candidate_depth':ct['depth']})
 finally:c.quit();b.quit();o.quit()
 s={'positions':len(rows),'equal_node':{'baseline_mean_regret_cp':mean(rows,'equal_node_baseline_regret'),'candidate_mean_regret_cp':mean(rows,'equal_node_candidate_regret'),'baseline_mean_ms':mean(rows,'equal_node_baseline_ms'),'candidate_mean_ms':mean(rows,'equal_node_candidate_ms')},'fixed_time':{'baseline_mean_regret_cp':mean(rows,'fixed_time_baseline_regret'),'candidate_mean_regret_cp':mean(rows,'fixed_time_candidate_regret'),'baseline_mean_nodes':mean(rows,'fixed_time_baseline_nodes'),'candidate_mean_nodes':mean(rows,'fixed_time_candidate_nodes'),'baseline_mean_depth':mean(rows,'fixed_time_baseline_depth'),'candidate_mean_depth':mean(rows,'fixed_time_candidate_depth')}}
 s['equal_node']['regret_improvement_cp']=s['equal_node']['baseline_mean_regret_cp']-s['equal_node']['candidate_mean_regret_cp'];s['fixed_time']['regret_improvement_cp']=s['fixed_time']['baseline_mean_regret_cp']-s['fixed_time']['candidate_mean_regret_cp'];out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'rows.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
if __name__=='__main__':main()
PY

echo
echo '=== ZERO-GAME POSITION TEST ==='
python "$ROOT/test_positions.py" --candidate "$CAND_EXE" --baseline "$BASE_EXE" --output "$RESULTS" --nodes "$NODE_BUDGET" --time-ms "$TIME_MS" --oracle-nodes "$ORACLE_NODES" --grade-nodes "$GRADE_NODES" | tee "$RESULTS/run.log"
echo
echo '=== COMPLETE ==='; echo "Results: $RESULTS"; cat "$RESULTS/summary.json"; echo; echo 'NO GAMES PLAYED. NO SOURCE MODIFIED. NO PUSH. NO MERGE.'
'@

$Runner = $Runner.Replace('__CAND_REPO__',$CandidateRepo).Replace('__CAND_REF__',$CandidateRef).Replace('__SF_REPO__',$StockfishRepo).Replace('__SF_SHA__',$StockfishSha).Replace('__NODES__',"$NodeBudget").Replace('__TIME_MS__',"$TimeMs").Replace('__ORACLE__',"$OracleNodes").Replace('__GRADE__',"$GradeNodes").Replace('__RESULTS__',$ResultUnix)
$Runner = $Runner -replace "`r`n","`n"
[System.IO.File]::WriteAllText($RunnerWin,$Runner,[System.Text.UTF8Encoding]::new($false))
$EscapedRunner = $RunnerWin.Replace("'", "'\''")
$RunnerUnix = (& $Bash -lc "cygpath -u '$EscapedRunner'").Trim()

Write-Host ''
Write-Host 'Starting native Windows/MSYS2 test. First run may take several minutes because MSYS2 packages and NNUE files are downloaded.' -ForegroundColor Green
& $Bash -lc "bash '$RunnerUnix'"
if ($LASTEXITCODE -ne 0) { throw "Hardware test failed with exit code $LASTEXITCODE. See $Results for logs." }

Write-Host ''
Write-Host 'DONE.' -ForegroundColor Green
Write-Host "Results: $Results"
$Summary = Join-Path $Results 'summary.json'
if (Test-Path $Summary) { Write-Host ''; Write-Host 'summary.json:' -ForegroundColor Cyan; Get-Content $Summary }
