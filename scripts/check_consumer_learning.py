#!/usr/bin/env python3
"""One real-base graft-training smoke test, not a reasoning-capability promotion.

Protocol is fixed before evaluation. The 64 protected test passages are never
used by the optimizer or replay. This does not exclude inherited pretraining
contamination and is not standard concatenated WikiText perplexity.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

DATASET_REVISION = '00aa25585682d4957f9e86edc73f59be7419af99'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'evidence/consumer/learning')
    parser.add_argument('--steps', type=int, default=80)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import transformers
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from leviathan.consumer.profiles import get_profile
    from leviathan.consumer.recurrence import NRDFConfig, QwenNRDFWrapper, install_nrdf, graft_parameters
    from leviathan.consumer.runtime import save_graft
    from leviathan.consumer.training import sample_depth
    from check_consumer_hf import arc_smoke

    torch.manual_seed(917)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(917)
    profile = get_profile('rtx3060')
    tokenizer = AutoTokenizer.from_pretrained(profile.repo_id, revision=profile.revision,
                                             trust_remote_code=False)
    files = {split: hf_hub_download('Salesforce/wikitext',
             filename=f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet',
             repo_type='dataset', revision=DATASET_REVISION) for split in ('train', 'test')}
    datasets = load_dataset('parquet', data_files=files)
    def passages(split, count):
        result, seen = [], set()
        for index, row in enumerate(datasets[split]):
            text = row['text']
            if len(text.strip()) < 160 or text.strip().startswith('='):
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)[:128]
            if len(ids) < 32:
                continue
            token_hash = sha256(json.dumps(ids).encode()).hexdigest()
            if token_hash in seen:
                continue
            seen.add(token_hash)
            result.append({'row': index, 'text_hash': sha256(text.encode()).hexdigest(),
                           'token_hash': token_hash, 'ids': ids})
            if len(result) == count:
                return result
        raise ValueError('Insufficient unique passages')
    train_pool, protected = passages('train', 128), passages('test', 64)
    training, replay = train_pool[:64], train_pool[64:]
    for field in ('text_hash', 'token_hash'):
        if {p[field] for p in train_pool} & {p[field] for p in protected}:
            raise ValueError('Protected passage overlaps training/replay')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'one actual 1.7B CPU BF16 graft-training smoke test',
        'profile': profile.as_dict(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'dataset': 'Salesforce/wikitext',
        'dataset_revision': DATASET_REVISION, 'config': 'wikitext-2-raw-v1',
        'seed': 917, 'steps': args.steps, 'max_tokens_per_passage': 128,
        'heldout_protocol': 'first 64 non-heading test rows >=160 chars and >=32 tokens; truncate to 128',
        'heldout_is_standard_wikitext_perplexity': False,
        'excluded_from_optimizer_and_replay': True,
        'inherited_pretraining_contamination': 'unknown',
        'manifest': {key: [{k: v for k, v in p.items() if k != 'ids'} for p in values]
                     for key, values in [('train', training), ('replay', replay), ('test', protected)]},
        'promoted': False, 'gpu_executed': False, 'nf4_executed': False}
    def write_report():
        (args.output / 'learning-report.json').write_text(json.dumps(report, indent=2))
    write_report()
    model = Qwen3ForCausalLM.from_pretrained(profile.repo_id, revision=profile.revision,
        torch_dtype=torch.bfloat16, device_map={'': 'cpu'}, low_cpu_mem_usage=True,
        attn_implementation='eager', trust_remote_code=False).eval()
    profile.validate_config(model.config.to_dict())
    def batch(p):
        ids = torch.tensor([p['ids']], dtype=torch.long)
        return {'input_ids': ids, 'labels': ids, 'use_cache': False}
    def evaluate():
        model.eval()
        total, tokens = 0., 0
        started = time.perf_counter()
        losses = []
        with torch.inference_mode():
            for i, p in enumerate(protected):
                n = len(p['ids']) - 1
                loss = float(model(**batch(p)).loss)
                if not torch.isfinite(torch.tensor(loss)):
                    raise FloatingPointError('Nonfinite heldout loss')
                total += loss * n
                tokens += n
                losses.append(loss)
        return {'mean_token_nll': total / tokens, 'predicted_tokens': tokens,
                'passages': len(protected), 'seconds': time.perf_counter() - started,
                'passage_losses': losses}
    report['baseline'] = evaluate()
    print('Baseline heldout: ' + json.dumps(report['baseline']), flush=True)
    report['arc_easy_baseline'] = arc_smoke(model, tokenizer, 8)
    original = list(model.parameters())
    for p in original:
        p.requires_grad_(False)
    versions = [(p, p._version) for p in original]
    paths = install_nrdf(model, NRDFConfig(latent_dim=32, heads=4, slots=4,
                                          min_loops=1, max_loops=4, chunk_tokens=32))
    report['graft_paths'] = paths
    params = list(graft_parameters(model))
    assert not ({id(p) for p in params} & {id(p) for p in original})
    report['trainable_graft_parameters'] = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    report['optimizer'] = {'name': 'AdamW', 'learning_rate': 1e-3, 'gradient_clip': 1.,
                           'depths': [1, 2, 3, 4], 'data': 'alternating train/replay'}
    report['history'] = []
    write_report()
    model.train()
    start = time.perf_counter()
    for step in range(args.steps):
        source = training if step % 2 == 0 else replay
        p = source[int(torch.randint(len(source), (), generator=generator))]
        depth = sample_depth(1, 4, generator)
        for m in model.modules():
            if isinstance(m, QwenNRDFWrapper):
                m.loops = depth
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch(p)).loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss; experiment aborted')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        if (step + 1) % 10 == 0 or step + 1 == args.steps:
            entry = {'step': step + 1, 'depth': depth, 'loss': float(loss.detach())}
            report['history'].append(entry)
            print('Training: ' + json.dumps(entry), flush=True)
    report['training_seconds'] = time.perf_counter() - start
    report['donor_gradient_isolation_pass'] = all(p.grad is None for p in original)
    report['donor_version_counters_unchanged'] = all(p._version == v for p, v in versions)
    assert report['donor_gradient_isolation_pass'] and report['donor_version_counters_unchanged']
    report['graft_influences'] = {}
    for name, module in model.named_modules():
        if isinstance(module, QwenNRDFWrapper):
            module.loops = 4
            report['graft_influences'][name] = float(module.gate.detach().tanh())
    report['candidate'] = evaluate()
    report['relative_heldout_nll_change'] = report['candidate']['mean_token_nll'] / report['baseline']['mean_token_nll'] - 1
    report['heldout_loss_gate_pass'] = report['relative_heldout_nll_change'] <= .02
    report['arc_easy_candidate'] = arc_smoke(model, tokenizer, 8)
    baseline_arc, candidate_arc = report['arc_easy_baseline'], report['arc_easy_candidate']
    report['arc_smoke_no_regression'] = (candidate_arc['correct'] >= baseline_arc['correct'] and
                                        candidate_arc['token_normalized_correct'] >= baseline_arc['token_normalized_correct'])
    report['decision'] = ('candidate_only_broad_tests_pending' if report['heldout_loss_gate_pass'] and
                          report['arc_smoke_no_regression'] else 'reject_candidate_for_promotion')
    report['missing_promotion_evidence'] = ['full ARC-Easy and Challenge', 'MMLU', 'GSM8K', 'coding',
        'calibration', 'safety', 'matched GPU efficiency', 'replicated training seeds']
    save_graft(model, args.output / 'graft', profile)
    write_report()
    print('FINAL: ' + json.dumps({k: v for k, v in report.items() if k not in {'manifest', 'history'}}), flush=True)

if __name__ == '__main__':
    main()
