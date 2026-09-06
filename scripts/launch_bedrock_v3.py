#!/usr/bin/env python3
"""Minimal supervisor: fixed environment, visible output, bounded no-output stalls."""
import json,os,queue,signal,subprocess,sys,threading,time,webbrowser
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    output=ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'scripts/run_bedrock_v3.py'),'--output',str(output)]
    print('No installs, no drive scan, no training. Reusing v7 CUDA and cached Qwen.',flush=True)
    print('ARC-Easy results are saved as each question finishes.',flush=True)
    kwargs={'start_new_session':True} if os.name!='nt' else {}
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
        encoding='utf-8',errors='replace',bufsize=1,**kwargs)
    messages=queue.Queue()
    def read():
        for line in process.stdout:messages.put(line)
        messages.put(None)
    threading.Thread(target=read,daemon=True).start()
    last=time.monotonic();heartbeat=last;done=False;timeout=False
    with (output/'CONSOLE.txt').open('w',encoding='utf-8') as log:
        while not done:
            now=time.monotonic()
            try:line=messages.get(timeout=1)
            except queue.Empty:line=''
            if line is None:done=True
            elif line:
                print(line,end='',flush=True);log.write(line);log.flush();last=time.monotonic()
            if now-heartbeat>=15:
                print(f'Working; {int(now-last)} seconds since the last completed operation.',flush=True)
                heartbeat=now
            if now-last>180:
                timeout=True
                print('STOPPED: child produced no progress for 180 seconds. Partial results kept.',flush=True)
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                else:os.killpg(process.pid,signal.SIGTERM)
                break
    try:code=process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();code=2
    if timeout:
        (output/'STALL.txt').write_text('No child output for 180 seconds; this is NOT a completed benchmark.',encoding='utf-8')
        # Do not leave an old success label on a timed-out run.
        path=output/'RESULTS.json'
        if path.exists():
            data=json.loads(path.read_text(encoding='utf-8'));data['status']='stopped_no_progress';data['error']='No child output for 180 seconds; benchmark incomplete'
            path.write_text(json.dumps(data,indent=2),encoding='utf-8')
            from run_bedrock_v3 import render
            (output/'RESULTS.html').write_text(render(data),encoding='utf-8')
        code=2
    target=output/'RESULTS.html'
    if target.exists():webbrowser.open(target.resolve().as_uri())
    else:webbrowser.open((output/'CONSOLE.txt').resolve().as_uri())
    print(f'Results and log: {output}',flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
