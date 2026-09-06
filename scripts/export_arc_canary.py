#!/usr/bin/env python3
"""Freeze 50 ARC-Easy examples. Cache first; optional small official-data download.

Not called with network permission by the one-click launcher. CI may export the
small data file once, then it is shipped for a completely offline benchmark.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OFFICIAL_SHA256='4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177'
URL='https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/test-00000-of-00001.parquet?download=true'


def export(output:Path,*,allow_download=False,limit=50):
    rows=None;source={}
    if allow_download:
        request=urllib.request.Request(URL,headers={'User-Agent':'Leviathan-no-training-eval/3'})
        with urllib.request.urlopen(request,timeout=30) as response:
            data=response.read(2*1024*1024)
        digest=hashlib.sha256(data).hexdigest()
        if digest!=OFFICIAL_SHA256:raise ValueError('Official ARC test bytes changed; refusing an unpinned dataset')
        import pyarrow.parquet as pq
        import pyarrow as pa
        rows=pq.read_table(pa.BufferReader(data)).to_pylist()[:limit]
        source={'official_parquet_sha256':digest,'source_url':URL}
    else:
        from datasets import Dataset
        cache=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'datasets'
        # Restrict lookups to the already-known dataset cache, not disks or user files.
        paths=sorted(cache.glob('allenai___ai2_arc/ARC-Easy/*/*/*test.arrow'))
        if not paths:paths=sorted(cache.glob('ai2_arc/ARC-Easy/*/*/*test.arrow'))
        for path in paths:
            ds=Dataset.from_file(str(path))
            if len(ds)>=limit and {'id','question','choices','answerKey'}<=set(ds.column_names):
                rows=[ds[i] for i in range(limit)]
                source={'cache_file':str(path),'dataset_fingerprint':ds._fingerprint};break
        if rows is None:
            raise RuntimeError('ARC-Easy is absent from the known local cache and bundle; no network request was made')
    if len(rows)!=limit:raise ValueError('Incomplete ARC canary')
    if rows[0]['id']!='Mercury_417466':raise ValueError('ARC ordering differs from the previous canary')
    encoded=json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    report={'dataset':'allenai/ai2_arc','config':'ARC-Easy','split':'test','selection':f'first {limit}',
        'license':'CC-BY-SA-4.0','attribution':'AI2 Reasoning Challenge (ARC), Allen Institute for AI',
        'official_source':'https://huggingface.co/datasets/allenai/ai2_arc',
        'license_url':'https://creativecommons.org/licenses/by-sa/4.0/',
        'modification':'First test examples exported to JSON; contents unchanged.',
        'examples_sha256':hashlib.sha256(encoded).hexdigest(),'source':source,'examples':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Exported {limit} ARC-Easy examples; SHA256={report["examples_sha256"]}',flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/arc_easy_50.json')
    p.add_argument('--allow-download',action='store_true');args=p.parse_args()
    export(args.output,allow_download=args.allow_download)
