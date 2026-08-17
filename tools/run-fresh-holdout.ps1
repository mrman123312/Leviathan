$ErrorActionPreference='Stop'
$Root=Join-Path $HOME 'LeviathanHardwareResults'
$src=Get-ChildItem $Root -Directory | Sort-Object LastWriteTime -Descending | Where-Object { (Test-Path (Join-Path $_.FullName 'stockfish-baseline.exe')) -and (Test-Path (Join-Path $_.FullName 'leviathan-candidate.exe')) } | Select-Object -First 1
if(-not $src){ throw "No completed local hardware build found under $Root" }
$base=Join-Path $src.FullName 'stockfish-baseline.exe'; $cand=Join-Path $src.FullName 'leviathan-candidate.exe'
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'; $out=Join-Path $Root ("fresh-holdout-"+$stamp); New-Item -ItemType Directory -Force -Path $out | Out-Null
$py='C:\msys64\ucrt64\bin\python.exe'; if(-not (Test-Path $py)){ throw 'MSYS2 UCRT64 Python not found.' }
$deps=Join-Path $out 'pydeps'; & $py -m pip install --disable-pip-version-check --target $deps chess | Out-Host; if($LASTEXITCODE-ne 0){throw 'python-chess install failed'}
$seed=Get-Random -Minimum 1 -Maximum 2147483646; Set-Content -Path (Join-Path $out 'seed.txt') -Value $seed
$test=Join-Path $out 'fresh_holdout.py'
@'
import argparse,json,random,statistics,time
from pathlib import Path
import chess,chess.engine

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

def cp(info,color):
 s=info.get('score'); return None if s is None else s.pov(color).score(mate_score=100000)

def choose(e,b,limit,token):
 t=time.perf_counter(); r=e.play(b,limit,info=chess.engine.INFO_ALL,game=token)
 return dict(move=r.move,nodes=r.info.get('nodes'),depth=r.info.get('depth'),seldepth=r.info.get('seldepth'),ms=(time.perf_counter()-t)*1000)

def grade(e,b,m,n,token):
 if m is None:return None
 info=e.analyse(b,chess.engine.Limit(nodes=n),root_moves=[m],game=token)
 return cp(info,b.turn)

def boot(vals,rng,B=4000):
 if not vals:return [None,None]
 means=[]; n=len(vals)
 for _ in range(B): means.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
 means.sort(); return [means[int(.025*B)],means[int(.975*B)-1]]

def summarize(rows,prefix,rng):
 d=[r[f'{prefix}_base_regret']-r[f'{prefix}_cand_regret'] for r in rows]
 ratios=[]
 if prefix=='node': ratios=[r['node_cand_ms']/r['node_base_ms'] for r in rows if r['node_base_ms']>0]
 else: ratios=[r['time_cand_nodes']/r['time_base_nodes'] for r in rows if r['time_base_nodes']]
 ds=sorted(d)
 trim=ds[2:-2] if len(ds)>4 else ds
 return {
  'positions':len(d),'base_mean_regret_cp':statistics.mean(r[f'{prefix}_base_regret'] for r in rows),
  'cand_mean_regret_cp':statistics.mean(r[f'{prefix}_cand_regret'] for r in rows),'mean_improvement_cp':statistics.mean(d),
  'median_improvement_cp':statistics.median(d),'trimmed_mean_improvement_cp':statistics.mean(trim),
  'bootstrap95_mean_improvement_cp':boot(d,rng),'better':sum(x>0 for x in d),'tie':sum(x==0 for x in d),'worse':sum(x<0 for x in d),
  'largest_rescue_cp':max(d),'largest_harm_cp':min(d),'median_ratio_candidate_over_stockfish':statistics.median(ratios)
 }

def main(a):
 rng=random.Random(a.seed); base=chess.engine.SimpleEngine.popen_uci(a.base); cand=chess.engine.SimpleEngine.popen_uci(a.cand); oracle=chess.engine.SimpleEngine.popen_uci(a.base)
 for e in (base,cand,oracle):cfg(e)
 # Warmup
 for i,e in enumerate((base,cand,oracle)): e.analyse(chess.Board(),chess.engine.Limit(nodes=10000),game=('warm',i))
 positions=[]; seen=set(); attempts=0
 try:
  while len(positions)<a.positions and attempts<600:
   attempts+=1; b=chess.Board(); target=rng.choice([12,16,20,24,28])
   ok=True
   for ply in range(target):
    if b.is_game_over(): ok=False; break
    infos=base.analyse(b,chess.engine.Limit(nodes=1200),multipv=min(4,b.legal_moves.count()),game=('gen',attempts,ply))
    if not isinstance(infos,list): infos=[infos]
    choices=[x['pv'][0] for x in infos if x.get('pv')]
    if not choices: ok=False; break
    b.push(rng.choice(choices))
   if not ok or b.is_game_over() or b.legal_moves.count()<8: continue
   f=b.fen(); key=' '.join(f.split()[:4])
   if key in seen:continue
   fi=base.analyse(b,chess.engine.Limit(nodes=12000),game=('filter',attempts)); s=cp(fi,b.turn)
   if s is None or abs(s)>180:continue
   seen.add(key); positions.append(f); print(f'generated {len(positions)}/{a.positions}: ply={b.ply()} eval={s}',flush=True)
  if len(positions)<a.positions: raise RuntimeError(f'only generated {len(positions)} positions')
  Path(a.out,'fens.txt').write_text('\n'.join(positions)+'\n')
  rows=[]
  for i,fen in enumerate(positions):
   b=chess.Board(fen); color=b.turn; print(f'test {i+1}/{len(positions)}',flush=True)
   oi=oracle.analyse(b,chess.engine.Limit(nodes=a.oracle),game=('oracle',i)); os=cp(oi,color); om=oi.get('pv',[None])[0] if oi.get('pv') else None
   # Alternate engine order to reduce thermal/order bias
   if i%2==0:
    bn=choose(base,b,chess.engine.Limit(nodes=a.nodes),('bn',i)); cn=choose(cand,b,chess.engine.Limit(nodes=a.nodes),('cn',i))
   else:
    cn=choose(cand,b,chess.engine.Limit(nodes=a.nodes),('cn',i)); bn=choose(base,b,chess.engine.Limit(nodes=a.nodes),('bn',i))
   bg=grade(oracle,b,bn['move'],a.grade,('bg',i)); cg=grade(oracle,b,cn['move'],a.grade,('cg',i))
   if i%2==0:
    bt=choose(base,b,chess.engine.Limit(time=a.ms/1000),('bt',i)); ct=choose(cand,b,chess.engine.Limit(time=a.ms/1000),('ct',i))
   else:
    ct=choose(cand,b,chess.engine.Limit(time=a.ms/1000),('ct',i)); bt=choose(base,b,chess.engine.Limit(time=a.ms/1000),('bt',i))
   btg=grade(oracle,b,bt['move'],a.grade,('btg',i)); ctg=grade(oracle,b,ct['move'],a.grade,('ctg',i))
   reg=lambda g:max(0,os-g) if g is not None and os is not None else None
   rows.append({'fen':fen,'oracle_move':str(om),'oracle_score':os,'node_base_move':str(bn['move']),'node_cand_move':str(cn['move']),'node_base_regret':reg(bg),'node_cand_regret':reg(cg),'node_base_ms':bn['ms'],'node_cand_ms':cn['ms'],'time_base_move':str(bt['move']),'time_cand_move':str(ct['move']),'time_base_regret':reg(btg),'time_cand_regret':reg(ctg),'time_base_nodes':bt['nodes'],'time_cand_nodes':ct['nodes'],'time_base_depth':bt['depth'],'time_cand_depth':ct['depth']})
  br=random.Random(a.seed^0x5A17)
  summary={'seed':a.seed,'positions':len(rows),'equal_node':summarize(rows,'node',br),'fixed_time':summarize(rows,'time',br)}
  Path(a.out,'rows.json').write_text(json.dumps(rows,indent=2)); Path(a.out,'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
 finally:
  cand.quit(); base.quit(); oracle.quit()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--cand',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--positions',type=int,default=48);p.add_argument('--nodes',type=int,default=100000);p.add_argument('--ms',type=int,default=100);p.add_argument('--oracle',type=int,default=1200000);p.add_argument('--grade',type=int,default=400000);main(p.parse_args())
'@ | Set-Content -Encoding utf8 $test
$old=$env:PYTHONPATH; $env:PYTHONPATH=$deps
Write-Host "Using existing binaries from $($src.FullName)" -ForegroundColor Cyan
Write-Host "Fresh holdout seed: $seed" -ForegroundColor Cyan
Write-Host 'Games: 0' -ForegroundColor Cyan
& $py $test --base $base --cand $cand --out $out --seed $seed --positions 48 --nodes 100000 --ms 100 --oracle 1200000 --grade 400000
$env:PYTHONPATH=$old
if($LASTEXITCODE-ne 0){throw "Fresh holdout failed. See $out"}
Write-Host "`nDONE: $out" -ForegroundColor Green
Get-Content (Join-Path $out 'summary.json')