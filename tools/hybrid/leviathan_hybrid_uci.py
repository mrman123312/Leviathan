#!/usr/bin/env python3
"""Leviathan Hybrid UCI proxy: CPU alpha-beta + asynchronous GPU advisor.

Uses authorized UCI ponder time as a speculative compute window. The CPU engine
remains authoritative; GPU output only decides where speculative work is spent.
"""
from __future__ import annotations
import argparse, json, queue, subprocess, sys, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
try:
    from gpu_risk_model import GpuRiskScorer, ReplyFeatures
except ImportError:
    from .gpu_risk_model import GpuRiskScorer, ReplyFeatures

INFO_KEYS={"depth","seldepth","multipv","nodes","nps","hashfull","time"}
def norm(s:str)->str:return " ".join(s.strip().split())
def append_move_to_position(position_cmd:str,move:str)->str:
    cmd=norm(position_cmd)
    if not cmd.startswith("position "): raise ValueError(f"not a UCI position command: {position_cmd!r}")
    return f"{cmd} {move}" if " moves " in f" {cmd} " else f"{cmd} moves {move}"
def strip_ponder(go_cmd:str)->str:return " ".join(p for p in norm(go_cmd).split() if p!="ponder")
def parse_setoption(line:str)->Tuple[str,Optional[str]]:
    p=norm(line).split()
    if len(p)<3 or p[:2]!=["setoption","name"]:return "",None
    try:i=p.index("value",2)
    except ValueError:return " ".join(p[2:]),None
    return " ".join(p[2:i])," ".join(p[i+1:])
def parse_bestmove(line:str)->Tuple[Optional[str],Optional[str]]:
    p=norm(line).split()
    if len(p)<2 or p[0]!="bestmove":return None,None
    ponder=None
    if "ponder" in p:
        i=p.index("ponder");ponder=p[i+1] if i+1<len(p) else None
    return p[1],ponder
def parse_info(line:str)->Optional[dict]:
    p=norm(line).split()
    if not p or p[0]!="info":return None
    out={"depth":0,"seldepth":0,"multipv":1,"nodes":0,"nps":0,"hashfull":0,"time":0,"score_cp":0,"score_mate":None,"pv":[]};i=1
    while i<len(p):
        t=p[i]
        if t in INFO_KEYS and i+1<len(p):
            try:out[t]=int(p[i+1])
            except ValueError:pass
            i+=2;continue
        if t=="score" and i+2<len(p):
            kind=p[i+1]
            try:v=int(p[i+2])
            except ValueError:v=0
            if kind=="cp":out["score_cp"]=v
            elif kind=="mate":out["score_mate"]=v;out["score_cp"]=(100000-min(abs(v),999)*100)*(1 if v>0 else -1)
            i+=3
            if i<len(p) and p[i] in ("lowerbound","upperbound"):i+=1
            continue
        if t=="pv":out["pv"]=p[i+1:];break
        i+=1
    return out
def allocate_integer_budget(total:int,weights:Sequence[float],minimum:int=1)->List[int]:
    n=len(weights)
    if not n:return []
    if total<n*minimum:return [1 if i<total else 0 for i in range(n)]
    base=[minimum]*n;remaining=total-n*minimum
    if remaining<=0:return base
    safe=[max(0.,float(w)) for w in weights];s=sum(safe)
    if s<=0:safe=[1.]*n;s=float(n)
    raw=[remaining*w/s for w in safe];floors=[int(x) for x in raw];out=[b+f for b,f in zip(base,floors)];left=total-sum(out)
    order=sorted(range(n),key=lambda i:raw[i]-floors[i],reverse=True)
    for i in order[:left]:out[i]+=1
    return out

class EngineProcess:
    def __init__(self,path:str,label:str):
        self.path=path;self.label=label
        self.proc=subprocess.Popen([path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1,universal_newlines=True)
        if self.proc.stdin is None or self.proc.stdout is None:raise RuntimeError(f"failed pipes for {label}")
        self.q:queue.Queue[str]=queue.Queue();self._write_lock=threading.Lock();self.initialized=False;self.options={}
        threading.Thread(target=self._reader_loop,daemon=True,name=f"{label}-stdout").start()
    def _reader_loop(self):
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:self.q.put(line.rstrip("\r\n"))
        finally:self.q.put("__LEV_ENGINE_EOF__")
    def send(self,line:str):
        if self.proc.poll() is not None:raise RuntimeError(f"engine {self.label} exited {self.proc.returncode}")
        with self._write_lock:
            assert self.proc.stdin is not None;self.proc.stdin.write(line.rstrip()+"\n");self.proc.stdin.flush()
    def read(self,timeout=None):return self.q.get(timeout=timeout)
    def wait_for(self,prefix:str,timeout:float=10.,collect=None):
        end=time.monotonic()+timeout
        while True:
            left=end-time.monotonic()
            if left<=0:raise TimeoutError(f"{self.label}: waiting for {prefix}")
            line=self.read(left)
            if line=="__LEV_ENGINE_EOF__":raise RuntimeError(f"{self.label}: exited waiting for {prefix}")
            if collect is not None:collect.append(line)
            if line.startswith(prefix):return line
    def drain(self):
        out=[]
        while True:
            try:l=self.q.get_nowait()
            except queue.Empty:break
            if l!="__LEV_ENGINE_EOF__":out.append(l)
        return out
    def close(self):
        try:
            if self.proc.poll() is None:self.send("quit");self.proc.wait(timeout=1.)
        except Exception:
            try:self.proc.kill()
            except Exception:pass

@dataclass
class ReplyCandidate:
    move:str;rank:int;score_cp:int;depth:int;seldepth:int;nodes:int;nps:int;hashfull:int;pv_len:int;mate_flag:int;predicted:bool=False;reply_probability:float=0.;risk:float=0.;utility:float=0.
@dataclass
class ScoutAssignment:
    candidate:ReplyCandidate;engine:EngineProcess;position_cmd:str;threads:int;hash_mb:int;started_at:float;stopped:bool=False
@dataclass
class PonderSession:
    generation:int;after_our_move_cmd:str;predicted_position_cmd:str;predicted_reply:Optional[str];go_ponder_cmd:str;started_at:float;cancel:threading.Event=field(default_factory=threading.Event);assignments:Dict[str,ScoutAssignment]=field(default_factory=dict);candidates:List[ReplyCandidate]=field(default_factory=list);ready:threading.Event=field(default_factory=threading.Event);gui_stop_seen:bool=False

class HybridProxy:
    HYBRID_PREFIX="Leviathan Hybrid"
    def __init__(self,args):
        self.args=args;self.primary=EngineProcess(args.engine,"primary");self.predictor=None;self.spare_engines=[];self.child_options={}
        self.stdout_lock=threading.Lock();self.state_lock=threading.RLock();self.log_lock=threading.Lock();self.predictor_lock=threading.Lock();self.pool_lock=threading.Lock();self.log_path=Path(args.log) if args.log else None
        self.hybrid_enabled=True;self.max_scouts=max(1,args.max_scouts);self.reply_nodes=max(100,args.reply_nodes);self.full_threads=max(1,args.threads);self.full_hash=max(1,args.hash);self.scout_hash=max(1,args.scout_hash);self.gpu_device=args.gpu_device;self.model_path=args.model;self.risk_weight=max(0.,args.risk_weight);self.scorer=GpuRiskScorer(self.gpu_device,self.model_path)
        self.current_position_cmd=None;self.current_search_position_cmd=None;self.last_own_search_position_cmd=None;self.last_bestmove=None;self.last_ponder_move=None;self.after_our_move_cmd=None;self.session=None;self.warm_match=None;self.generation=0;self.foreground_engine=None;self.foreground_relay=None;self.closed=False;self.prewarm_started=False
    def emit(self,line):
        with self.stdout_lock:sys.stdout.write(line+"\n");sys.stdout.flush()
    def log(self,event,**payload):
        if self.log_path is None:return
        rec={"ts":time.time(),"event":event,**payload};self.log_path.parent.mkdir(parents=True,exist_ok=True)
        with self.log_lock:
            with self.log_path.open("a",encoding="utf-8") as f:f.write(json.dumps(rec,sort_keys=True)+"\n")
    def init_engine(self,e,copy_options=True):
        if e.initialized:return
        e.send("uci");e.wait_for("uciok",20);e.initialized=True
        if copy_options:
            for n,v in self.child_options.items():
                if n in ("Threads","Hash","MultiPV"):continue
                e.send(f"setoption name {n}"+(f" value {v}" if v is not None else ""))
        e.send("isready");e.wait_for("readyok",20)
    def ensure_predictor(self):
        with self.predictor_lock:
            if self.predictor is None or self.predictor.proc.poll() is not None:
                self.predictor=EngineProcess(self.args.opponent_engine or self.args.engine,"opponent-predictor");self.init_engine(self.predictor,False)
            return self.predictor
    def ensure_spares(self,n):
        with self.pool_lock:
            while len(self.spare_engines)<n:
                e=EngineProcess(self.args.engine,f"scout-{len(self.spare_engines)+1}");self.init_engine(e,True);self.spare_engines.append(e)
            return self.spare_engines[:n]
    def set_engine_option(self,e,n,v):
        e.send(f"setoption name {n}"+(f" value {v}" if v is not None else ""));e.options[n]=v
    def configure_search_engine(self,e,threads,hash_mb,multipv=1):
        self.init_engine(e,True)
        if e.options.get("Threads")!=str(threads):self.set_engine_option(e,"Threads",str(threads))
        if hash_mb is not None and e.options.get("Hash")!=str(hash_mb):self.set_engine_option(e,"Hash",str(hash_mb))
        if e.options.get("MultiPV")!=str(multipv):self.set_engine_option(e,"MultiPV",str(multipv))
        e.send("isready");e.wait_for("readyok",20)
    def handle_uci(self):
        self.primary.send("uci")
        while True:
            line=self.primary.read(20)
            if line=="__LEV_ENGINE_EOF__":raise RuntimeError("primary exited during uci")
            if line=="uciok":
                self.emit("option name Leviathan Hybrid Enabled type check default true");self.emit(f"option name Leviathan Hybrid Scouts type spin default {self.max_scouts} min 1 max 8");self.emit(f"option name Leviathan Hybrid Reply Nodes type spin default {self.reply_nodes} min 100 max 5000000");self.emit(f"option name Leviathan Hybrid Scout Hash type spin default {self.scout_hash} min 1 max 1024");self.emit("option name Leviathan Hybrid GPU Device type combo default auto var auto var cuda var cpu var off");self.emit("option name Leviathan Hybrid Model type string default <empty>");self.emit(f"option name Leviathan Hybrid Risk Weight type spin default {int(self.risk_weight*100)} min 0 max 400");self.emit("uciok");self.primary.initialized=True;break
            self.emit(line)
    def handle_setoption(self,line):
        n,v=parse_setoption(line)
        if n.startswith(self.HYBRID_PREFIX):
            s=n[len(self.HYBRID_PREFIX):].strip()
            if s=="Enabled":self.hybrid_enabled=(v or "true").lower() in ("true","1","yes","on")
            elif s=="Scouts":self.max_scouts=max(1,min(8,int(v or self.max_scouts)))
            elif s=="Reply Nodes":self.reply_nodes=max(100,int(v or self.reply_nodes))
            elif s=="Scout Hash":self.scout_hash=max(1,int(v or self.scout_hash))
            elif s=="GPU Device":self.gpu_device=v or "auto";self.scorer=GpuRiskScorer(self.gpu_device,self.model_path)
            elif s=="Model":self.model_path=None if not v or v=="<empty>" else v;self.scorer=GpuRiskScorer(self.gpu_device,self.model_path)
            elif s=="Risk Weight":self.risk_weight=max(0.,float(int(v or 100))/100.)
            self.log("hybrid_option",name=n,value=v);return
        if n:
            self.child_options[n]=v
            if n=="Threads" and v is not None:self.full_threads=max(1,int(v))
            elif n=="Hash" and v is not None:self.full_hash=max(1,int(v))
        self.primary.send(line)
        for e in self.spare_engines:
            if e is not self.foreground_engine:
                try:e.send(line)
                except Exception:pass
    def handle_isready(self):
        self.primary.send("isready");self.primary.wait_for("readyok",20);self.emit("readyok")
        if self.hybrid_enabled and not self.prewarm_started:self.prewarm_started=True;threading.Thread(target=self.prewarm_workers,daemon=True,name="hybrid-prewarm").start()
    def prewarm_workers(self):
        try:self.ensure_predictor();self.ensure_spares(max(0,min(self.max_scouts,self.full_threads)-1));self.log("workers_prewarmed",scouts=len(self.spare_engines),predictor=True)
        except Exception as exc:self.log("prewarm_error",error=repr(exc))
    def handle_position(self,line):
        line=norm(line)
        with self.state_lock:
            self.current_position_cmd=line;self.warm_match=None;sess=self.session
            if sess is not None and sess.assignments:
                a=sess.assignments.get(line)
                if a is not None and not a.stopped:self.warm_match=a;self.log("ponder_hit_branch",generation=sess.generation,reply=a.candidate.move,rank=a.candidate.rank,utility=a.candidate.utility,harvest_ms=int((time.monotonic()-sess.started_at)*1000));return
                if sess.ready.is_set():self.log("ponder_miss",generation=sess.generation,position=line);self.cancel_session(None)
            self.primary.send(line)
    def handle_go(self,line):
        line=norm(line)
        if " ponder" in f" {line}" or line.startswith("go ponder"):self.start_ponder_session(line);return
        with self.state_lock:warm=self.warm_match;self.warm_match=None
        if warm is not None:self.promote_warm_scout(warm,line);return
        self.cancel_session(None);self.configure_search_engine(self.primary,self.full_threads,self.full_hash,1)
        if self.current_position_cmd:self.primary.send(self.current_position_cmd)
        self.current_search_position_cmd=self.current_position_cmd;self.last_own_search_position_cmd=self.current_position_cmd;self.primary.send(line);self.start_relay(self.primary)
    def handle_ponderhit(self):
        with self.state_lock:sess=self.session;current=self.current_position_cmd;a=sess.assignments.get(current) if sess and current else None
        if a is not None and not a.stopped:self.promote_warm_scout(a,strip_ponder(sess.go_ponder_cmd));return
        self.cancel_session(None)
        if self.current_position_cmd:self.primary.send(self.current_position_cmd)
        self.configure_search_engine(self.primary,self.full_threads,self.full_hash,1);self.current_search_position_cmd=self.current_position_cmd;self.last_own_search_position_cmd=self.current_position_cmd;self.primary.send(strip_ponder(sess.go_ponder_cmd) if sess else "go");self.start_relay(self.primary)
    def handle_stop(self):
        with self.state_lock:sess=self.session;fg=self.foreground_engine
        if fg is not None:
            try:fg.send("stop")
            except Exception:pass
            return
        if sess is not None:
            sess.gui_stop_seen=True;a=sess.assignments.get(norm(sess.predicted_position_cmd)) if sess.assignments else None
            if a is not None and not a.stopped:self.stop_assignment(a,True);return
            sess.cancel.set()
            if self.current_position_cmd:
                self.configure_search_engine(self.primary,1,None,1);self.primary.send(self.current_position_cmd);self.primary.send("go nodes 1")
                try:
                    while True:
                        o=self.primary.read(2)
                        if o.startswith("bestmove"):self.emit(o);break
                except Exception:self.emit("bestmove 0000")
                return
        self.primary.send("stop")
    def handle_ucinewgame(self):
        self.cancel_session(None);self.primary.send("ucinewgame")
        for e in self.spare_engines:
            try:e.send("ucinewgame")
            except Exception:pass
        self.current_position_cmd=self.current_search_position_cmd=self.last_own_search_position_cmd=self.last_bestmove=self.last_ponder_move=self.after_our_move_cmd=None
    def start_relay(self,e):
        with self.state_lock:self.foreground_engine=e
        self.foreground_relay=threading.Thread(target=self._relay_loop,args=(e,),daemon=True,name="uci-relay");self.foreground_relay.start()
    def _relay_loop(self,e):
        try:
            while True:
                line=e.read(3600)
                if line=="__LEV_ENGINE_EOF__":self.emit("bestmove 0000");break
                self.emit(line)
                if line.startswith("bestmove"):
                    best,ponder=parse_bestmove(line)
                    with self.state_lock:
                        self.last_bestmove=best;self.last_ponder_move=ponder
                        if self.current_search_position_cmd and best and best!="0000":
                            try:self.after_our_move_cmd=append_move_to_position(self.current_search_position_cmd,best)
                            except ValueError:self.after_our_move_cmd=None
                        self.foreground_engine=None
                    self.log("own_bestmove",bestmove=best,ponder=ponder,position=self.current_search_position_cmd);break
        except Exception as exc:self.log("relay_error",error=repr(exc),engine=e.label);self.emit("bestmove 0000");self.foreground_engine=None
    def start_ponder_session(self,go):
        if not self.hybrid_enabled or not self.after_our_move_cmd:
            self.configure_search_engine(self.primary,self.full_threads,self.full_hash,1)
            if self.current_position_cmd:self.primary.send(self.current_position_cmd)
            self.primary.send(go);self.generation+=1;self.session=PonderSession(self.generation,self.after_our_move_cmd or self.current_position_cmd or "position startpos",self.current_position_cmd or "position startpos",self.last_ponder_move,go,time.monotonic());return
        self.cancel_session(None);self.generation+=1;s=PonderSession(self.generation,self.after_our_move_cmd,self.current_position_cmd or append_move_to_position(self.after_our_move_cmd,self.last_ponder_move or "0000"),self.last_ponder_move,go,time.monotonic());self.session=s;self.log("ponder_start",generation=s.generation,after_our_move=s.after_our_move_cmd,predicted_reply=s.predicted_reply,threads=self.full_threads,gpu=self.scorer.describe());threading.Thread(target=self._prepare_multi_ponder,args=(s,),daemon=True,name=f"ponder-{s.generation}").start()
    def _prepare_multi_ponder(self,s):
        try:
            k=max(1,min(self.max_scouts,self.full_threads));cands=self.generate_reply_candidates(s,max(k+2,k))
            if s.cancel.is_set() or self.session is not s:return
            if not cands:
                if s.predicted_reply:cands=[ReplyCandidate(s.predicted_reply,1,0,0,0,0,0,0,0,0,True)]
                else:return
            if s.predicted_reply and all(c.move!=s.predicted_reply for c in cands):
                best=cands[0].score_cp if cands else 0;cands.append(ReplyCandidate(s.predicted_reply,len(cands)+1,best-10,0,0,0,0,0,0,0,True))
            for c in cands:c.predicted=c.predicted or c.move==s.predicted_reply
            best=max(c.score_cp for c in cands);features=[ReplyFeatures(c.rank,c.score_cp,max(0,best-c.score_cp),c.depth,c.seldepth,c.nodes,c.nps,c.hashfull,c.pv_len,1. if c.predicted else 0.,float(c.mate_flag)) for c in cands];scores=self.scorer.score(features)
            for c,x in zip(cands,scores):c.reply_probability=float(x["reply_probability"]);c.risk=float(x["risk"]);c.utility=c.reply_probability*(1.+self.risk_weight*c.risk)
            cands.sort(key=lambda c:(c.utility,c.predicted,-c.rank),reverse=True);selected=cands[:k]
            if s.predicted_reply and all(c.move!=s.predicted_reply for c in selected):selected[-1]=next(c for c in cands if c.move==s.predicted_reply)
            selected.sort(key=lambda c:c.utility,reverse=True);alloc=allocate_integer_budget(self.full_threads,[c.utility for c in selected],1);engines=[self.primary]+self.ensure_spares(max(0,len(selected)-1));assign={}
            for cand,threads,e in zip(selected,alloc,engines):
                if s.cancel.is_set() or self.session is not s:return
                h=self.full_hash if e is self.primary else min(self.full_hash,self.scout_hash);self.configure_search_engine(e,max(1,threads),h,1);pos=append_move_to_position(s.after_our_move_cmd,cand.move);e.drain();e.send(pos);e.send(s.go_ponder_cmd);assign[norm(pos)]=ScoutAssignment(cand,e,norm(pos),max(1,threads),h,time.monotonic())
            with self.state_lock:
                if s.cancel.is_set() or self.session is not s:return
                s.candidates=selected;s.assignments=assign;s.ready.set()
            self.log("ponder_pool_ready",generation=s.generation,setup_ms=int((time.monotonic()-s.started_at)*1000),candidates=[{"move":c.move,"rank":c.rank,"score_cp":c.score_cp,"depth":c.depth,"seldepth":c.seldepth,"nodes":c.nodes,"nps":c.nps,"hashfull":c.hashfull,"pv_len":c.pv_len,"mate_flag":c.mate_flag,"reply_probability":c.reply_probability,"risk":c.risk,"utility":c.utility,"threads":alloc[i],"predicted":c.predicted} for i,c in enumerate(selected)],gpu=self.scorer.describe())
        except Exception as exc:self.log("ponder_prepare_error",generation=s.generation,error=repr(exc));s.ready.set()
    def generate_reply_candidates(self,s,multipv):
        with self.predictor_lock:
            if self.predictor is None or self.predictor.proc.poll() is not None:self.predictor=EngineProcess(self.args.opponent_engine or self.args.engine,"opponent-predictor");self.init_engine(self.predictor,False)
            e=self.predictor;self.configure_search_engine(e,1,min(32,self.full_hash),multipv);e.drain();e.send(s.after_our_move_cmd);e.send(f"go nodes {self.reply_nodes}");latest={};end=time.monotonic()+max(2.,self.args.predictor_timeout)
            while time.monotonic()<end and not s.cancel.is_set():
                try:line=e.read(.1)
                except queue.Empty:continue
                if line=="__LEV_ENGINE_EOF__":break
                info=parse_info(line)
                if info and info.get("pv"):
                    m=int(info.get("multipv",1));prev=latest.get(m)
                    if prev is None or (info.get("depth",0),info.get("nodes",0))>=(prev.get("depth",0),prev.get("nodes",0)):latest[m]=info
                if line.startswith("bestmove"):break
            else:
                try:e.send("stop")
                except Exception:pass
        out=[];seen=set()
        for m in sorted(latest):
            i=latest[m];pv=i.get("pv") or []
            if not pv or pv[0] in seen:continue
            move=pv[0];seen.add(move);out.append(ReplyCandidate(move,m,int(i.get("score_cp",0)),int(i.get("depth",0)),int(i.get("seldepth",0)),int(i.get("nodes",0)),int(i.get("nps",0)),int(i.get("hashfull",0)),len(pv),1 if i.get("score_mate") is not None else 0,move==s.predicted_reply))
        return out
    def stop_assignment(self,a,emit_bestmove=False):
        if a.stopped:return None
        a.stopped=True
        try:a.engine.send("stop")
        except Exception:return None
        end=time.monotonic()+3.;best=None
        while time.monotonic()<end:
            try:l=a.engine.read(.1)
            except queue.Empty:continue
            if l.startswith("bestmove"):best=l;self.emit(l) if emit_bestmove else None;break
        return best
    def promote_warm_scout(self,a,go):
        s=self.session;harvest=int((time.monotonic()-(s.started_at if s else a.started_at))*1000);self.stop_assignment(a,False);selected=a.engine;old=self.primary
        if selected is not old:
            self.primary=selected
            for x in list(s.assignments.values()) if s else []:
                if x.engine is old and not x.stopped:self.stop_assignment(x,False)
            if selected in self.spare_engines:self.spare_engines.remove(selected)
            if old not in self.spare_engines:self.spare_engines.insert(0,old)
        self.cancel_session(selected);self.configure_search_engine(self.primary,self.full_threads,None,1);self.primary.send(a.position_cmd);self.current_position_cmd=a.position_cmd;self.current_search_position_cmd=a.position_cmd;self.last_own_search_position_cmd=a.position_cmd;self.primary.send(strip_ponder(go));self.log("warm_promote",reply=a.candidate.move,rank=a.candidate.rank,probability=a.candidate.reply_probability,risk=a.candidate.risk,utility=a.candidate.utility,ponder_threads=a.threads,promoted_threads=self.full_threads,harvest_ms=harvest,engine=a.engine.label);self.start_relay(self.primary)
    def cancel_session(self,keep):
        with self.state_lock:s=self.session;self.session=None;self.warm_match=None
        if s is None:return
        s.cancel.set()
        for a in list(s.assignments.values()):
            if a.engine is keep:continue
            if not a.stopped:self.stop_assignment(a,False)
    def run(self):
        try:
            for raw in sys.stdin:
                line=norm(raw)
                if not line:continue
                try:
                    if line=="uci":self.handle_uci()
                    elif line.startswith("setoption name "):self.handle_setoption(line)
                    elif line=="isready":self.handle_isready()
                    elif line=="ucinewgame":self.handle_ucinewgame()
                    elif line.startswith("position "):self.handle_position(line)
                    elif line.startswith("go ") or line=="go":self.handle_go(line)
                    elif line=="ponderhit":self.handle_ponderhit()
                    elif line=="stop":self.handle_stop()
                    elif line=="quit":break
                    else:self.primary.send(line)
                except Exception as exc:self.log("command_error",command=line,error=repr(exc));self.emit("bestmove 0000") if line.startswith("go") else None
        finally:self.close()
        return 0
    def close(self):
        if self.closed:return
        self.closed=True;self.cancel_session(None);engines=[self.primary]+self.spare_engines+([self.predictor] if self.predictor else []);seen=set()
        for e in engines:
            if e is None or id(e) in seen:continue
            seen.add(id(e));e.close()

def build_arg_parser():
    p=argparse.ArgumentParser(description="Leviathan CPU+GPU multi-ponder UCI proxy");p.add_argument("--engine",required=True);p.add_argument("--opponent-engine",default=None);p.add_argument("--model",default=None);p.add_argument("--gpu-device",default="auto",choices=("auto","cuda","cpu","off"));p.add_argument("--max-scouts",type=int,default=4);p.add_argument("--reply-nodes",type=int,default=12000);p.add_argument("--predictor-timeout",type=float,default=3.);p.add_argument("--threads",type=int,default=1);p.add_argument("--hash",type=int,default=64);p.add_argument("--scout-hash",type=int,default=32);p.add_argument("--risk-weight",type=float,default=1.);p.add_argument("--log",default="local_results/hybrid/session.jsonl");return p
def main():return HybridProxy(build_arg_parser().parse_args()).run()
if __name__=="__main__":raise SystemExit(main())
