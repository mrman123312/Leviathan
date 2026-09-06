#!/usr/bin/env python3
"""Build the same offline label-separated ARC data from a PINNED official checkout."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('data/arc_agi_1'))
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    manifest={'repository':'fchollet/ARC-AGI','revision':'399030444e0ab0cc8b4e199870fb20b863846f34','license':'Apache-2.0','files':{},'protocol':'test outputs separated; original repository allows 3 attempts; this harness reports stricter pass@1 and pass@2'}
    for split in ('training','evaluation'):
        tasks={};labels={}
        for p in sorted((args.upstream/'data'/split).glob('*.json')):
            task=json.loads(p.read_text());labels[p.stem]=[r['output'] for r in task['test']]
            tasks[p.stem]={'train':task['train'],'test':[{'input':r['input']} for r in task['test']]}
        if len(tasks)!=400:raise ValueError('Expected 400 tasks in each pinned split')
        for name,obj in ((split+'_tasks.json',tasks),(split+'_labels.json',labels)):
            raw=json.dumps(obj,separators=(',',':')).encode();(root/name).write_bytes(raw)
            manifest['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'tasks':len(obj)}
    shutil.copyfile(args.upstream/'LICENSE',root/'LICENSE')
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
