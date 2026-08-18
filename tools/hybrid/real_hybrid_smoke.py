#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path

def send(p,s):p.stdin.write(s+'\n');p.stdin.flush()
def read_until(p,prefix,timeout=30.):
    end=time.monotonic()+timeout;lines=[]
    while time.monotonic()<end:
        line=p.stdout.readline()
        if not line:
            if p.poll() is not None:raise RuntimeError(f'proxy exited {p.returncode}')
            continue
        line=line.strip();lines.append(line)
        if line.startswith(prefix):return line,lines
    raise TimeoutError(f'{prefix}: {lines[-20:]}')
def parts(line):
    p=line.split();best=p[1] if len(p)>1 else None;ponder=None
    if 'ponder' in p:
        i=p.index('ponder');ponder=p[i+1] if i+1<len(p) else None
    return best,ponder
def main():
    a=argparse.ArgumentParser();a.add_argument('--engine',required=True);a.add_argument('--opponent-engine',default=None);a.add_argument('--device',default='cuda');a.add_argument('--threads',type=int,default=4);a.add_argument('--hash',type=int,default=64);a.add_argument('--ponder-seconds',type=float,default=2.);a.add_argument('--log',default='local_results/hybrid/real-smoke.jsonl');x=a.parse_args();wrapper=Path(__file__).resolve().parent/'leviathan_hybrid_uci.py';cmd=[sys.executable,str(wrapper),'--engine',x.engine,'--gpu-device',x.device,'--threads',str(x.threads),'--hash',str(x.hash),'--max-scouts',str(min(4,x.threads)),'--log',x.log]
    if x.opponent_engine:cmd+=['--opponent-engine',x.opponent_engine]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    try:
        send(p,'uci');read_until(p,'uciok');send(p,f'setoption name Threads value {x.threads}');send(p,f'setoption name Hash value {x.hash}');send(p,'setoption name Ponder value true');send(p,'isready');read_until(p,'readyok');send(p,'position startpos');send(p,'go nodes 50000');bm,_=read_until(p,'bestmove',60);best,ponder=parts(bm)
        if not best or not ponder:raise RuntimeError(f'no ponder move: {bm}')
        send(p,f'position startpos moves {best} {ponder}');send(p,'go ponder wtime 60000 btime 60000 winc 0 binc 0');time.sleep(x.ponder_seconds);send(p,'ponderhit');bm2,_=read_until(p,'bestmove',60);send(p,'quit');p.wait(timeout=5);print(json.dumps({'first_bestmove':bm,'ponder_result':bm2,'log':x.log,'returncode':p.returncode},indent=2));text=Path(x.log).read_text(encoding='utf-8')
        if '"event": "ponder_pool_ready"' not in text or '"event": "warm_promote"' not in text:raise SystemExit('hybrid path did not reach pool-ready + warm-promote')
        return 0
    finally:
        if p.poll() is None:p.kill()
if __name__=='__main__':raise SystemExit(main())
