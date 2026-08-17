#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,os,random,re,statistics,subprocess
from pathlib import Path

def uci(o):
    return ''.join(f"setoption name {k} value {str(v).lower() if isinstance(v,bool) else v}\n" for k,v in o.items())
def normalized(out):
    rows=[]
    for raw in out.splitlines():
        line=raw.strip()
        if line.startswith('Position:') or line.startswith('bestmove '): rows.append(line)
        elif line.startswith('info ') and not line.startswith('info string '):
            line=re.sub(r'\s+nps\s+\d+','',line);line=re.sub(r'\s+time\s+\d+','',line);rows.append(line)
    return '\n'.join(rows)
def run(bin_,opts,cmd='bench'):
    p=subprocess.run([bin_],input=uci(opts)+cmd+'\nquit\n',text=True,capture_output=True,check=True);o=p.stdout+p.stderr;n=normalized(o)
    return {'ms':int(re.findall(r'Total time \(ms\)\s*:\s*(\d+)',o)[-1]),'nodes':int(re.findall(r'Nodes searched\s*:\s*(\d+)',o)[-1]),'sig':hashlib.sha256(n.encode()).hexdigest(),'lines':len(n.splitlines())}
def boot_ci(xs,seed,n=25000):
    r=random.Random(seed);m=len(xs);ys=[statistics.median(xs[r.randrange(m)] for _ in range(m)) for _ in range(n)];ys.sort();return [ys[int(.025*n)],ys[int(.975*n)]]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True);ap.add_argument('--aa',required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--options',type=Path,required=True);ap.add_argument('--rounds',type=int,default=21);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();opts=json.loads(a.options.read_text())
    bins={'ref':a.reference,'aa':a.aa,'candidate':a.candidate};cmds={'default':'bench','depth11':'bench 16 1 11 default depth','nodes50k':'bench 16 1 50000 default nodes'}
    sig={k:{} for k in bins}
    for name,b in bins.items():
        for label,cmd in cmds.items():
            x=run(b,opts,cmd);sig[name][label]={k:x[k] for k in ('nodes','sig','lines')}
    if sig['aa']!=sig['ref'] or sig['candidate']!=sig['ref']: raise SystemExit(f'FUNCTIONAL DIVERGENCE {sig}')
    for b in bins.values(): run(b,opts)
    rng=random.Random(89102026);obs={'aa':[],'candidate':[]}
    # Both contenders are measured in exactly the same A-X-A geometry. Their order
    # inside each round is randomized, so calibration and candidate see the same
    # cache/thermal/order conditions instead of letting the A/A binary double as a reference.
    for i in range(a.rounds):
        order=['aa','candidate'];rng.shuffle(order)
        for who in order:
            before=run(a.reference,opts);mid=run(bins[who],opts);after=run(a.reference,opts)
            if (before['nodes'],before['sig'],before['lines'])!=(mid['nodes'],mid['sig'],mid['lines']) or (before['nodes'],before['sig'],before['lines'])!=(after['nodes'],after['sig'],after['lines']): raise SystemExit('DIVERGENCE DURING TIMING')
            obs[who].append(math.sqrt(before['ms']*after['ms'])/mid['ms'])
    res={}
    for j,who in enumerate(('aa','candidate')):
        xs=obs[who];med=statistics.median(xs);ci=boot_ci(xs,9100+j);fast=sum(x>1 for x in xs)
        res[who]={'rounds':len(xs),'median_speedup':med,'mean_speedup':statistics.fmean(xs),'bootstrap_median_95pct_ci':ci,'faster_rounds':fast}
    aa=res['aa'];cand=res['candidate']
    aa_ok=abs(aa['median_speedup']-1)<=.004 and aa['bootstrap_median_95pct_ci'][0]<=1<=aa['bootstrap_median_95pct_ci'][1]
    cand['status']='PROVISIONAL_WIN' if aa_ok and cand['bootstrap_median_95pct_ci'][0]>1.002 and cand['faster_rounds']>=math.ceil(a.rounds*.75) else ('INVALID_CALIBRATION' if not aa_ok else 'RETEST_INCONCLUSIVE')
    aa['status']='CALIBRATION_PASS' if aa_ok else 'CALIBRATION_FAIL'
    payload={'schema':'LV_LOSSLESS_PAIR_V2','runner':{'image':os.environ.get('ImageOS'),'version':os.environ.get('ImageVersion')},'signatures':sig,'results':res};a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
    if not aa_ok: raise SystemExit('A/A calibration failed')
if __name__=='__main__':main()
