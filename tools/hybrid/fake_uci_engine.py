#!/usr/bin/env python3
from __future__ import annotations
import sys,threading,time
position="position startpos";threads=1;multipv=1;stop_flag=threading.Event()
def out(s):print(s,flush=True)
def root_moves():
    if "e2e4" in position and "e7e5" not in position and "c7c5" not in position:return [("e7e5",28),("c7c5",21),("g8f6",5),("d7d5",-15),("e7e6",-25)]
    if "e7e5" in position or "c7c5" in position or "g8f6" in position:return [("g1f3",34),("f1c4",19),("d2d4",10)]
    return [("e2e4",30),("d2d4",20),("g1f3",8)]
def run_search(cmd):
    stop_flag.clear();is_ponder="ponder" in cmd.split();moves=root_moves();mpv=min(multipv,len(moves))
    for depth in (6,8,10):
        if stop_flag.is_set():break
        for i in range(mpv):
            move,cp=moves[i];out(f"info depth {depth} seldepth {depth+4+i} multipv {i+1} score cp {cp} nodes {depth*1000+i*100} nps 500000 hashfull 10 time {depth*2} pv {move} a7a6")
        time.sleep(.01)
    if is_ponder:
        while not stop_flag.wait(.01):pass
    best=moves[0][0];ponder="b8c6" if best=="g1f3" else ("e7e5" if best=="e2e4" else "g1f3");out(f"bestmove {best} ponder {ponder}")
def launch(cmd):threading.Thread(target=run_search,args=(cmd,),daemon=True).start()
for raw in sys.stdin:
    line=" ".join(raw.strip().split())
    if not line:continue
    if line=="uci":
        out("id name Leviathan Fake");out("id author test");out("option name Threads type spin default 1 min 1 max 128");out("option name Hash type spin default 16 min 1 max 65536");out("option name MultiPV type spin default 1 min 1 max 10");out("option name Ponder type check default false");out("uciok")
    elif line.startswith("setoption name "):
        p=line.split()
        try:i=p.index("value");name=" ".join(p[2:i]);value=" ".join(p[i+1:])
        except ValueError:name=" ".join(p[2:]);value=""
        if name=="Threads":threads=int(value)
        elif name=="MultiPV":multipv=int(value)
    elif line=="isready":out("readyok")
    elif line.startswith("position "):position=line
    elif line.startswith("go"):launch(line)
    elif line in ("ponderhit","stop"):stop_flag.set()
    elif line=="quit":stop_flag.set();break
