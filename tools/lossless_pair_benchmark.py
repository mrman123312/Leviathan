#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,os,random,re,statistics,subprocess
from pathlib import Path

def uci(options):
    out=[]
    for k,v in options.items():
        if isinstance(v,bool):v=str(v).lower()
        out.append(f"setoption name {k} value {v}")
    return "\n".join(out)+"\n"

def norm(o):
    xs=[]
    for raw in o.splitlines():
        line=raw.strip()
        if line.startswith('Position:') or line.startswith('bestmove '):xs.append(line)
        elif line.startswith('info ') and not line.startswith('info string '):
            line=re.sub(r'\s+nps\s+\d+','',line);line=re.sub(r'\s+time\s+\d+','',line);xs.append(line)
    return '\n'.join(xs)

def run(binary,options,command='bench'):
    p=subprocess.run([binary],input=uci(options)+command+'\nquit\n',text=True,capture_output=True,check=True);o=p.stdout+p.stderr;t=norm(o)
    return {'ms':int(re.findall(r'Total time \(ms\)\s*:\s*(\d+)',o)[-1]),'nodes':int(re.findall(r'Nodes searched\s*:\s*(\d+)',o)[-1]),'nps':int(re.findall(r'Nodes/second\s*:\s*(\d+)',o)[-1]),'behavior_sha256':hashlib.sha256(t.encode()).hexdigest(),'behavior_lines':len(t.splitlines())}

def ci(vals,seed,samples=30000):
    rng=random.Random(seed);n=len(vals);m=[statistics.median(vals[rng.randrange(n)] for _ in range(n)) for _ in range(samples)];m.sort();return [m[int(samples*.025)],m[int(samples*.975)]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--control-a',required=True);ap.add_argument('--control-b',required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--options',type=Path,required=True);ap.add_argument('--rounds',type=int,default=31);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();opts=json.loads(a.options.read_text());eng={'control_a':a.control_a,'control_b':a.control_b,'candidate':a.candidate}
    cmds={'default':'bench','depth11':'bench 16 1 11 default depth','nodes50k':'bench 16 1 50000 default nodes'};keys=('nodes','behavior_sha256','behavior_lines');sig={n:{} for n in eng}
    for n,b in eng.items():
        for label,cmd in cmds.items():
            r=run(b,opts,cmd);sig[n][label]={k:r[k] for k in keys}
    ref=sig['control_a'];div={n:v for n,v in sig.items() if v!=ref}
    if div:raise SystemExit(f'FUNCTIONAL DIVERGENCE reference={ref} divergent={div}')
    for b in eng.values():run(b,opts)
    obs={'control_b':[],'candidate':[]};rng=random.Random(2026081604)
    for i in range(a.rounds):
        order=['control_b','candidate'];rng.shuffle(order)
        if i%2:order.reverse()
        for n in order:
            first='control_a' if i%2==0 else 'control_b';second='control_b' if i%2==0 else 'control_a';before=run(eng[first],opts);mid=run(eng[n],opts);after=run(eng[second],opts)
            if not all(before[k]==mid[k]==after[k] for k in keys):raise SystemExit('FUNCTIONAL DIVERGENCE during timing')
            obs[n].append({'round':i,'reference_first':first,'reference_first_ms':before['ms'],'candidate_ms':mid['ms'],'reference_second':second,'reference_second_ms':after['ms'],'sandwich_speedup':math.sqrt(before['ms']*after['ms'])/mid['ms']})
    results={}
    for ix,(n,rows) in enumerate(obs.items()):
        rs=[x['sandwich_speedup'] for x in rows];med=statistics.median(rs);interval=ci(rs,41000+ix);fast=sum(x>1 for x in rs)
        status=('CALIBRATION_PASS' if abs(med-1)<=.004 and interval[0]<=1<=interval[1] else 'CALIBRATION_FAIL') if n=='control_b' else ('PROVISIONAL_WIN' if interval[0]>1.002 and fast>=math.ceil(a.rounds*.75) else ('REJECT_REGRESSION' if interval[1]<.998 else 'RETEST_INCONCLUSIVE'))
        results[n]={'rounds':a.rounds,'median_speedup':med,'mean_speedup':statistics.mean(rs),'geometric_mean_speedup':math.exp(statistics.mean(math.log(x) for x in rs)),'bootstrap_median_95pct_ci':interval,'faster_rounds':fast,'status':status,'observations':rows}
    if results['control_b']['status']!='CALIBRATION_PASS':results['candidate']['status']='INVALID_CALIBRATION'
    payload={'schema':'LV_LOSSLESS_PAIR_V1','runner':{'image':os.environ.get('ImageOS'),'image_version':os.environ.get('ImageVersion'),'arch':os.uname().machine},'signatures':sig,'binary_sha256':{n:hashlib.sha256(Path(b).read_bytes()).hexdigest() for n,b in eng.items()},'results':results};a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
    if results['control_b']['status']!='CALIBRATION_PASS':raise SystemExit('A/A calibration failed')
if __name__=='__main__':main()
