#!/usr/bin/env python3
"""Native-HF integration; optional actual 1.7B checkpoint test (never a 27B proxy)."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from leviathan.consumer.profiles import get_profile
from leviathan.consumer.recurrence import NRDFConfig, QwenNRDFWrapper, install_nrdf, restore_nrdf, graft_parameters
from leviathan.consumer.runtime import save_graft, load_graft


def compare(model, batch, config, *, observe=True):
    model.eval()
    with torch.inference_mode():
        baseline = model(**batch, use_cache=False).logits.detach()
    paths = install_nrdf(model, config)
    model.eval()
    for module in model.modules():
        if isinstance(module, QwenNRDFWrapper):
            module.observe_at_zero = observe
    with torch.inference_mode():
        candidate = model(**batch, use_cache=False).logits
    error = float((candidate.float() - baseline.float()).abs().max())
    if error != 0:
        raise AssertionError(f"Zero gate logit drift: {error}")
    return {"max_abs_logit_difference": error, "argmax_all_positions": bool(
        (baseline.argmax(-1) == candidate.argmax(-1)).all()), "graft_paths": paths}


def arc_smoke(model, tokenizer, limit=8):
    from datasets import load_dataset
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    records = []
    def score(question, choice):
        context = f"Question: {question}\nAnswer:"
        prefix = tokenizer.encode(context, add_special_tokens=False)
        full = tokenizer.encode(context + " " + choice, add_special_tokens=False)
        if full[:len(prefix)] != prefix:
            raise ValueError("Tokenizer boundary changed; refusing misaligned option scores")
        ids = torch.tensor([full], device=model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits[0, len(prefix)-1:-1].float()
            labels = ids[0, len(prefix):]
            nll = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        return -float(nll), len(labels)
    start = time.perf_counter()
    for item in dataset.select(range(limit)):
        scores = [score(item["question"], c) for c in item["choices"]["text"]]
        predicted = max(range(len(scores)), key=lambda i: scores[i][0])
        normalized = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])
        labels = item["choices"]["label"]
        records.append({"id": item["id"], "prediction": labels[predicted],
                        "token_normalized_prediction": labels[normalized], "gold": item["answerKey"],
                        "correct": labels[predicted] == item["answerKey"],
                        "token_normalized_correct": labels[normalized] == item["answerKey"]})
        print(f"ARC-Easy canary {len(records)}/{limit}", flush=True)
    return {"scope": "first test examples, smoke only, NOT full ARC-Easy or lm-eval-equivalent",
            "protocol": "Question/Answer raw completion; sum/token-normalized option log probabilities",
            "dataset_fingerprint": dataset._fingerprint,
            "n": len(records), "correct": sum(r["correct"] for r in records),
            "token_normalized_correct": sum(r["token_normalized_correct"] for r in records),
            "seconds": time.perf_counter() - start, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-base", action="store_true", help="Download pinned 3.44 GB base, run CPU BF16")
    parser.add_argument("--arc-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/consumer/hf.json")
    args = parser.parse_args()
    torch.manual_seed(69)
    torch.set_num_threads(2)
    import transformers
    from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer, AutoConfig
    profile = get_profile("rtx3060")
    report = {"torch": torch.__version__, "transformers": transformers.__version__,
              "device": "cpu", "quantization": "none", "actual_pretrained": args.actual_base}
    for name in ("rtx3060", "qwen27b"):
        p = get_profile(name)
        config = AutoConfig.from_pretrained(p.repo_id, revision=p.revision, trust_remote_code=False)
        p.validate_config(config.to_dict())
        report[name + "_config_verified"] = p.revision
    if args.actual_base:
        tokenizer = AutoTokenizer.from_pretrained(profile.repo_id, revision=profile.revision, trust_remote_code=False)
        model = Qwen3ForCausalLM.from_pretrained(profile.repo_id, revision=profile.revision,
            torch_dtype=torch.bfloat16, device_map={"": "cpu"}, low_cpu_mem_usage=True,
            attn_implementation="eager", trust_remote_code=False).eval()
        text = "The capital of France is"
        batch = tokenizer(text, return_tensors="pt")
        report["profile"] = profile.as_dict()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=12, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        report["prompt"] = {"input": text, "completion": tokenizer.decode(
            output[0, batch.input_ids.shape[1]:], skip_special_tokens=True),
            "elapsed_seconds": time.perf_counter() - started,
            "generated_tokens": output.shape[-1] - batch.input_ids.shape[-1]}
    else:
        tiny = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                           max_position_embeddings=128)
        tiny._attn_implementation = "eager"
        model = Qwen3ForCausalLM(tiny).eval()
        batch = {"input_ids": torch.tensor([[1, 5, 7, 2, 8], [1, 9, 3, 4, 8]])}
    cfg = NRDFConfig(latent_dim=16, slots=3, heads=2, max_loops=3,
                     cell_width=128 if args.actual_base else 16, ancestral_cells=True)
    report["parity"] = compare(model, batch, cfg)
    if not args.actual_base:
        wrapper = next(m for m in model.modules() if isinstance(m, QwenNRDFWrapper))
        wrapper.set_influence(.2, experimental=True)
        wrapper.fabric.cell_gate.data.fill_(.2)
        for param in model.parameters():
            param.requires_grad_(False)
        for param in graft_parameters(model):
            param.requires_grad_(True)
        model.train()
        model(**batch, labels=batch["input_ids"], use_cache=False).loss.backward()
        assert wrapper.fabric.attention.in_proj_weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in wrapper.donor.parameters())
        model.eval()
        with torch.inference_mode():
            full = model(**batch, use_cache=False).logits
            pre = model(input_ids=batch["input_ids"][:, :-1], use_cache=True)
            tail = model(input_ids=batch["input_ids"][:, -1:], past_key_values=pre.past_key_values,
                         use_cache=True).logits
        torch.testing.assert_close(full[:, -1:], tail, atol=1e-5, rtol=1e-5)
        report["opened_graft_native_cache_parity"] = float((full[:, -1:] - tail).abs().max())
        report["donor_gradients_absent"] = True
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            save_graft(model, tmp, profile)
            expected = full.detach()
            restore_nrdf(model)
            load_graft(model, tmp, profile, allow_experimental=True)
            model.eval()
            with torch.inference_mode():
                reloaded = model(**batch, use_cache=False).logits
            torch.testing.assert_close(reloaded, expected, atol=0, rtol=0)
        report["graft_save_restore_exact"] = True
        from leviathan.consumer.speculation import same_model_generate
        one = batch["input_ids"][:1]
        for m in model.modules():
            if isinstance(m, QwenNRDFWrapper):
                m.loops = 3
        with torch.inference_mode():
            direct = one
            for _ in range(7):
                nxt = model(input_ids=direct, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                direct = torch.cat((direct, nxt), -1)
        spec, counters = same_model_generate(model, one, max_new_tokens=7,
                                             target_depth=3, draft_depth=1)
        assert torch.equal(direct, spec)
        report["same_model_speculative_greedy_exact"] = counters
    if args.actual_base and args.arc_limit:
        wrappers = [m for m in model.modules() if isinstance(m, QwenNRDFWrapper)]
        for m in wrappers:
            m.enabled = False
        report["arc_easy_baseline"] = arc_smoke(model, tokenizer, args.arc_limit)
        for m in wrappers:
            m.enabled = True
        report["arc_easy_zero_graft"] = arc_smoke(model, tokenizer, args.arc_limit)
        assert report["arc_easy_baseline"]["records"] == report["arc_easy_zero_graft"]["records"]
    report["full_27b_executed"] = False
    report["rtx3060_executed"] = False
    report["nf4_executed"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
