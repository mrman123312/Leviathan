$ErrorActionPreference='Stop'

# Local three-way zero-game validation:
# pinned Stockfish vs P01 vs the EXACT historically confirmed P01+ProbCut-256 mutation.
# Reuses the user's existing Stockfish/P01 binaries and latest saved 48-position holdout.
# Builds ONLY the ProbCut candidate. Does not edit/push/merge any source repo and plays zero games.

$Root=Join-Path $HOME 'LeviathanHardwareResults'
$build=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object {
  (Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe')) -and
  (Test-Path (Join-Path $_.FullName 'leviathan-candidate.exe'))
} | Select-Object -First 1
if(-not $build){ throw "No completed Stockfish/P01 hardware build found under $Root" }

$hold=Get-ChildItem $Root -Directory -Filter 'fresh-holdout-*' | Sort-Object LastWriteTime -Descending | Where-Object {
  Test-Path (Join-Path $_.FullName 'fens.txt')
} | Select-Object -First 1
if(-not $hold){ throw "No saved fresh holdout with fens.txt found under $Root" }

$sf=Join-Path $build.FullName 'stockfish-baseline.exe'
$p01=Join-Path $build.FullName 'leviathan-candidate.exe'
$fens=Join-Path $hold.FullName 'fens.txt'

$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$out=Join-Path $Root ("probcut256-threeway-"+$stamp)
New-Item -ItemType Directory -Force -Path $out | Out-Null

$Bash='C:\msys64\usr\bin\bash.exe'
$Py='C:\msys64\ucrt64\bin\python.exe'
if(-not (Test-Path $Bash)){throw 'MSYS2 bash not found at C:\msys64\usr\bin\bash.exe'}
if(-not (Test-Path $Py)){throw 'MSYS2 UCRT64 Python not found.'}

function To-Msys([string]$p){
  $q=$p.Replace("'", "'\''")
  return (& $Bash -lc "cygpath -u '$q'").Trim()
}

$uOut=To-Msys $out
$uSF=To-Msys $sf
$uP01=To-Msys $p01
$uFens=To-Msys $fens

$deps=Join-Path $out 'pydeps'
& $Py -m pip install --disable-pip-version-check --target $deps chess | Out-Host
if($LASTEXITCODE-ne 0){throw 'python-chess install failed'}
$uDeps=To-Msys $deps

$runnerWin=Join-Path $env:TEMP 'leviathan-probcut256-build.sh'
$runner=@'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/ucrt64/bin:/usr/bin:$PATH
OUT="__OUT__"
ROOT="/tmp/leviathan-probcut256-$$"
SRC="$ROOT/candidate"
cleanup(){ rm -rf "$ROOT" || true; }
trap cleanup EXIT
mkdir -p "$ROOT" "$OUT"

echo '=== CLONE P01 FROM GITHUB ==='
git clone --filter=blob:none --no-checkout https://github.com/mrman123312/Leviathan.git "$SRC"
git -C "$SRC" fetch origin leviathan/fundamentals-ultra-p01-qfrontier
git -C "$SRC" checkout --detach FETCH_HEAD
BASE_SHA=$(git -C "$SRC" rev-parse HEAD)
echo "p01_base_sha=$BASE_SHA" | tee "$OUT/probcut-provenance.txt"

cat > "$ROOT/materialize.py" <<'PY'
import re,sys
from pathlib import Path
p=Path(sys.argv[1]); s=p.read_text()
pairs=[
 ('Move  move, excludedMove, bestMove;', 'Move  move, excludedMove, bestMove, probCutNearMiss;'),
 ('Value bestValue, value, eval, maxValue, probCutBeta;', 'Value bestValue, value, eval, maxValue, probCutBeta, probCutNearValue;'),
 ('bestMove       = Move::none();\n    priorReduction = (ss - 1)->reduction;',
  'bestMove         = Move::none();\n    probCutNearMiss  = Move::none();\n    probCutNearValue = -VALUE_INFINITE;\n    priorReduction   = (ss - 1)->reduction;'),
 ('MovePicker mp(pos, ttData.move, depth, &mainHistory, &lowPlyHistory, &captureHistory, contHist,\n                  &sharedHistory, ss->ply);',
  'const Move leviathanPreferredMove = ttData.move ? ttData.move : probCutNearMiss;\n    MovePicker mp(pos, leviathanPreferredMove, depth, &mainHistory, &lowPlyHistory, &captureHistory,\n                  contHist, &sharedHistory, ss->ply);')]
for a,b in pairs:
    n=s.count(a)
    if n!=1: raise SystemExit(f'anchor count={n} for {a[:70]!r}')
    s=s.replace(a,b,1)
pattern=r'(?ms)^(\s*)if \(value >= probCutBeta\)\n\1\{\n.*?^\1\}'
m=re.search(pattern,s)
if not m: raise SystemExit('ProbCut success-block regex anchor missing')
I=m.group(1)
extra=(f"\n{I}else if (value >= beta && !is_decisive(value) && value > probCutNearValue)\n"
       f"{I}{{\n{I}    probCutNearValue = value;\n{I}    probCutNearMiss  = move;\n{I}}}")
s=s[:m.end()]+extra+s[m.end():]
a='r += 697;  // Base reduction offset to compensate for other tweaks\n        r -= moveCount * 65;'
b='r += 697;  // Base reduction offset to compensate for other tweaks\n        if (move == probCutNearMiss)\n            r -= 256;\n        r -= moveCount * 65;'
if s.count(a)!=1: raise SystemExit('LMR base anchor missing')
s=s.replace(a,b,1)
p.write_text(s)
PY
python "$ROOT/materialize.py" "$SRC/src/search.cpp"
git -C "$SRC" diff --check
git -C "$SRC" diff -- src/search.cpp > "$OUT/probcut256.patch"
sha256sum "$OUT/probcut256.patch" | tee -a "$OUT/probcut-provenance.txt"

echo '=== BUILD EXACT P01+PROBCUT-256 ==='
cd "$SRC/src"
make net
make clean || true
if make -j"${NUMBER_OF_PROCESSORS:-4}" build ARCH=x86-64-avx2 COMP=mingw; then :
elif make -j"${NUMBER_OF_PROCESSORS:-4}" build ARCH=x86-64-avx2; then :
else
  echo 'AVX2 build failed; falling back to x86-64.'
  make clean || true
  make -j"${NUMBER_OF_PROCESSORS:-4}" build ARCH=x86-64 COMP=mingw
fi
exe=''
[[ -x stockfish.exe ]] && exe=stockfish.exe
[[ -z "$exe" && -x stockfish ]] && exe=stockfish
[[ -n "$exe" ]] || { echo 'candidate executable not found'; exit 2; }
cp "$exe" "$OUT/p01-probcut256.exe"
sha256sum "$OUT/p01-probcut256.exe" | tee -a "$OUT/probcut-provenance.txt"
echo 'PROBCUT256_BUILD_OK'
'@
$runner=$runner.Replace('__OUT__',$uOut) -replace "`r`n","`n"
[System.IO.File]::WriteAllText($runnerWin,$runner,[System.Text.UTF8Encoding]::new($false))
$uRunner=To-Msys $runnerWin
Write-Host "Building ONLY exact P01+ProbCut-256 from GitHub..." -ForegroundColor Cyan
& $Bash -lc "bash '$uRunner'"
if($LASTEXITCODE-ne 0){throw "ProbCut-256 build failed. See $out"}

$prob=Join-Path $out 'p01-probcut256.exe'
$uProb=To-Msys $prob

$test=Join-Path $out 'threeway.py'
@'
import argparse,json,random,statistics,time
from pathlib import Path
import chess,chess.engine

def cfg(e,lev=False):
 o={}
 if 'Threads' in e.options:o['Threads']=1
 if 'Hash' in e.options:o['Hash']=64
 if lev:
  if 'Leviathan Fundamentals' in e.options:o['Leviathan Fundamentals']=True
  if 'Leviathan Fundamentals Authority' in e.options:o['Leviathan Fundamentals Authority']=1
  if 'Leviathan Quiet Overdrive' in e.options:o['Leviathan Quiet Overdrive']=0
  for n in ['Leviathan Risk','Leviathan Policy','Leviathan MetaSearch','Leviathan Specialist','Leviathan Atlas','Leviathan Search DSL']:
   if n in e.options:o[n]=False
 if o:e.configure(o)

def cp(info,color):
 s=info.get('score'); return None if s is None else s.pov(color).score(mate_score=100000)

def choose(e,b,limit,token):
 t=time.perf_counter(); r=e.play(b,limit,info=chess.engine.INFO_ALL,game=token)
 return {'move':r.move,'nodes':r.info.get('nodes'),'depth':r.info.get('depth'),'seldepth':r.info.get('seldepth'),'ms':(time.perf_counter()-t)*1000}

def grade(e,b,m,n,token):
 if m is None:return None
 info=e.analyse(b,chess.engine.Limit(nodes=n),root_moves=[m],game=token)
 return cp(info,b.turn)

def boot(vals,rng,B=5000):
 if not vals:return [None,None]
 n=len(vals); z=[]
 for _ in range(B):z.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
 z.sort();return [z[int(.025*B)],z[int(.975*B)-1]]

def comp(rows,mode,a,b,rng):
 # Positive means a is better than b (b regret - a regret)
 d=[r[f'{mode}_{b}_regret']-r[f'{mode}_{a}_regret'] for r in rows]
 ds=sorted(d); trim=ds[2:-2] if len(ds)>4 else ds
 return {'mean_improvement_cp':statistics.mean(d),'median_improvement_cp':statistics.median(d),
         'trimmed_mean_improvement_cp':statistics.mean(trim),'bootstrap95_mean_improvement_cp':boot(d,rng),
         'better':sum(x>0 for x in d),'tie':sum(x==0 for x in d),'worse':sum(x<0 for x in d),
         'largest_rescue_cp':max(d),'largest_harm_cp':min(d)}

def main(a):
 fens=[x.strip() for x in Path(a.fens).read_text().splitlines() if x.strip()]
 sf=chess.engine.SimpleEngine.popen_uci(a.sf);p01=chess.engine.SimpleEngine.popen_uci(a.p01);prob=chess.engine.SimpleEngine.popen_uci(a.prob);oracle=chess.engine.SimpleEngine.popen_uci(a.sf)
 cfg(sf);cfg(oracle);cfg(p01,True);cfg(prob,True)
 eng={'sf':sf,'p01':p01,'prob':prob}
 for j,e in enumerate((sf,p01,prob,oracle)):e.analyse(chess.Board(),chess.engine.Limit(nodes=12000),game=('warm',j))
 rows=[]
 try:
  for i,fen in enumerate(fens):
   b=chess.Board(fen);print(f'threeway {i+1}/{len(fens)}',flush=True)
   oi=oracle.analyse(b,chess.engine.Limit(nodes=a.oracle),game=('oracle',i));os=cp(oi,b.turn);om=oi.get('pv',[None])[0] if oi.get('pv') else None
   row={'fen':fen,'oracle_score':os,'oracle_move':str(om)}
   order=['sf','p01','prob']; order=order[i%3:]+order[:i%3]
   chosen={}
   for name in order: chosen[('node',name)]=choose(eng[name],b,chess.engine.Limit(nodes=a.nodes),(f'node-{name}',i))
   order2=list(reversed(order))
   for name in order2: chosen[('time',name)]=choose(eng[name],b,chess.engine.Limit(time=a.ms/1000),(f'time-{name}',i))
   moves={x['move'] for x in chosen.values() if x['move'] is not None}
   grades={}
   for k,m in enumerate(sorted(moves,key=lambda x:x.uci())):grades[m]=grade(oracle,b,m,a.grade,('grade',i,k))
   for mode in ('node','time'):
    for name in ('sf','p01','prob'):
     x=chosen[(mode,name)];g=grades.get(x['move']);reg=None if os is None or g is None else max(0,os-g)
     row[f'{mode}_{name}_move']=str(x['move']);row[f'{mode}_{name}_regret']=reg;row[f'{mode}_{name}_nodes']=x['nodes'];row[f'{mode}_{name}_depth']=x['depth'];row[f'{mode}_{name}_ms']=x['ms']
   rows.append(row)
 finally:
  for e in (sf,p01,prob,oracle):e.quit()
 rng=random.Random(0x256BEEF)
 summary={'positions':len(rows),'source_fens':a.fens,
  'mean_regret_cp':{mode:{name:statistics.mean(r[f'{mode}_{name}_regret'] for r in rows) for name in ('sf','p01','prob')} for mode in ('node','time')},
  'equal_node':{'p01_vs_sf':comp(rows,'node','p01','sf',rng),'prob_vs_sf':comp(rows,'node','prob','sf',rng),'prob_vs_p01':comp(rows,'node','prob','p01',rng)},
  'fixed_time':{'p01_vs_sf':comp(rows,'time','p01','sf',rng),'prob_vs_sf':comp(rows,'time','prob','sf',rng),'prob_vs_p01':comp(rows,'time','prob','p01',rng)},
  'throughput':{
   'equal_node_median_time_ratio_p01_over_sf':statistics.median(r['node_p01_ms']/r['node_sf_ms'] for r in rows),
   'equal_node_median_time_ratio_prob_over_p01':statistics.median(r['node_prob_ms']/r['node_p01_ms'] for r in rows),
   'equal_node_median_time_ratio_prob_over_sf':statistics.median(r['node_prob_ms']/r['node_sf_ms'] for r in rows),
   'fixed_time_median_node_ratio_p01_over_sf':statistics.median(r['time_p01_nodes']/r['time_sf_nodes'] for r in rows if r['time_sf_nodes']),
   'fixed_time_median_node_ratio_prob_over_p01':statistics.median(r['time_prob_nodes']/r['time_p01_nodes'] for r in rows if r['time_p01_nodes']),
   'fixed_time_median_node_ratio_prob_over_sf':statistics.median(r['time_prob_nodes']/r['time_sf_nodes'] for r in rows if r['time_sf_nodes'])}}
 Path(a.out,'rows.json').write_text(json.dumps(rows,indent=2));Path(a.out,'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sf',required=True);p.add_argument('--p01',required=True);p.add_argument('--prob',required=True);p.add_argument('--fens',required=True);p.add_argument('--out',required=True);p.add_argument('--nodes',type=int,default=100000);p.add_argument('--ms',type=int,default=100);p.add_argument('--oracle',type=int,default=1200000);p.add_argument('--grade',type=int,default=400000);main(p.parse_args())
'@ | Set-Content -Encoding utf8 $test

$envOld=$env:PYTHONPATH;$env:PYTHONPATH=$deps
Write-Host "`nUsing saved holdout: $fens" -ForegroundColor Cyan
Write-Host 'Comparing: Stockfish vs P01 vs exact P01+ProbCut-256' -ForegroundColor Cyan
Write-Host 'Games: 0' -ForegroundColor Cyan
& $Py $test --sf $sf --p01 $p01 --prob $prob --fens $fens --out $out --nodes 100000 --ms 100 --oracle 1200000 --grade 400000
$env:PYTHONPATH=$envOld
if($LASTEXITCODE-ne 0){throw "Three-way test failed. See $out"}
Write-Host "`nDONE: $out" -ForegroundColor Green
Get-Content (Join-Path $out 'summary.json')
