#!/usr/bin/env python3
from __future__ import annotations
import math,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from gpu_risk_model import GpuRiskScorer,ReplyFeatures
from leviathan_hybrid_uci import allocate_integer_budget,append_move_to_position,parse_bestmove,parse_info,parse_setoption,strip_ponder

def read_until(p,prefix,timeout=8.):
    end=time.monotonic()+timeout;lines=[]
    while time.monotonic()<end:
        line=p.stdout.readline().strip()
        if line:lines.append(line)
        if line.startswith(prefix):return line,lines
    raise AssertionError(f"timeout {prefix}: {lines}")
def send(p,line):p.stdin.write(line+"\n");p.stdin.flush()
def wait_log(path,token,timeout=8.):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if path.exists() and token in path.read_text(encoding="utf-8"):return
        time.sleep(.05)
    raise AssertionError(path.read_text(encoding="utf-8") if path.exists() else token)
def spawn(log):
    fake=HERE/"fake_uci_engine.py";wrapper=HERE/"leviathan_hybrid_uci.py"
    return subprocess.Popen([sys.executable,str(wrapper),"--engine",str(fake),"--opponent-engine",str(fake),"--threads","4","--max-scouts","3","--reply-nodes","1000","--gpu-device","cpu","--log",str(log)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
def setup(p):
    send(p,"uci");_,lines=read_until(p,"uciok");assert any("Leviathan Hybrid Scouts" in x for x in lines);send(p,"setoption name Threads value 4");send(p,"setoption name Hash value 32");send(p,"isready");read_until(p,"readyok");time.sleep(1.2);send(p,"position startpos");send(p,"go nodes 2000");b,_=read_until(p,"bestmove");assert "e2e4" in b;send(p,"position startpos moves e2e4 e7e5");send(p,"go ponder wtime 60000 btime 60000 winc 0 binc 0")
def test_helpers():
    assert append_move_to_position("position startpos","e2e4")=="position startpos moves e2e4";assert strip_ponder("go ponder wtime 1")=="go wtime 1";assert parse_setoption("setoption name Threads value 8")==('Threads','8');assert parse_bestmove("bestmove e2e4 ponder e7e5")==('e2e4','e7e5');i=parse_info("info depth 12 seldepth 20 multipv 2 score cp 31 nodes 1000 nps 500000 hashfull 7 time 3 pv e7e5 g1f3");assert i and i['multipv']==2 and i['pv'][0]=='e7e5';assert allocate_integer_budget(8,[.6,.3,.1],1)==[4,3,1]
def test_gpu():
    s=GpuRiskScorer("cpu");o=s.score([ReplyFeatures(1,20,0,10,15,1000,500000,10,4,1),ReplyFeatures(2,-40,60,10,12,1000,500000,10,4,0)]);assert math.isclose(sum(x['reply_probability'] for x in o),1.,rel_tol=1e-6);assert o[0]['utility']>o[1]['utility']
def test_predicted():
    log=HERE/"_hybrid-smoke.jsonl";log.unlink(missing_ok=True);p=spawn(log)
    try:
        setup(p);wait_log(log,"ponder_pool_ready");send(p,"ponderhit");b,_=read_until(p,"bestmove");assert "g1f3" in b;wait_log(log,"warm_promote");send(p,"quit");p.wait(timeout=5)
    finally:
        if p.poll() is None:p.kill()
        log.unlink(missing_ok=True)
def test_alternative():
    log=HERE/"_hybrid-alt.jsonl";log.unlink(missing_ok=True);p=spawn(log)
    try:
        setup(p);wait_log(log,"ponder_pool_ready");send(p,"stop");read_until(p,"bestmove");send(p,"position startpos moves e2e4 c7c5");send(p,"go wtime 60000 btime 59000 winc 0 binc 0");b,_=read_until(p,"bestmove");assert "g1f3" in b;send(p,"quit");p.wait(timeout=5);t=log.read_text(encoding="utf-8");assert '"event": "warm_promote"' in t and '"reply": "c7c5"' in t
    finally:
        if p.poll() is None:p.kill()
        log.unlink(missing_ok=True)
if __name__=="__main__":test_helpers();test_gpu();test_predicted();test_alternative();print("hybrid core tests: PASS")
