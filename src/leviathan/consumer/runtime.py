"""Pinned one-model loading, genuine prompt execution, and explicit graft loading."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import time
import torch
from torch import nn
from .profiles import ModelProfile, get_profile
from .recurrence import NRDFConfig, QwenNRDFWrapper, install_nrdf


def load_model(profile: ModelProfile, *, quantization: str = "nf4", device: str = "cuda",
               local_files_only: bool = False, allow_large_model: bool = False):
    if quantization not in {"nf4", "none"} or device not in {"cuda", "cpu"}:
        raise ValueError("Use nf4/none and cuda/cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu --quantization none for CPU baseline")
    if quantization == "nf4" and device != "cuda":
        raise RuntimeError("This NF4 runtime is CUDA-only; no silent CPU offload or dequantization")
    if profile.parameter_estimate_b > 10 and not allow_large_model:
        raise RuntimeError("27B download/load needs --allow-large-model; 3060 test profile is the default")
    bits = 4 if quantization == "nf4" else (16 if device == "cuda" else 32)
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        nominal = profile.memory_estimate(bits)["nominal_weights_gib"] * 2**30
        if nominal + 2 * 2**30 > free:
            raise MemoryError("Nominal weights plus 2 GiB reserve exceed free VRAM. "
                              "Use rtx3060 profile; 27B is not resident on a 12 GB GPU.")
    try:
        import transformers
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install inference extras: python -m pip install -e '.[consumer]'") from exc
    common = {"revision": profile.revision, "local_files_only": local_files_only,
              "trust_remote_code": False}
    config = AutoConfig.from_pretrained(profile.repo_id, **common)
    profile.validate_config(config.to_dict())
    tokenizer = AutoTokenizer.from_pretrained(profile.repo_id, **common)
    model_class = getattr(transformers, profile.architecture, None)
    if model_class is None:
        raise RuntimeError(f"Installed Transformers lacks {profile.architecture}; upgrade inference extra")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32)
    options = dict(common, config=config, torch_dtype=dtype,
                   device_map={"": device}, low_cpu_mem_usage=True)
    if quantization == "nf4":
        try:
            import bitsandbytes
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes from consumer extras for NF4") from exc
        options["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    started = time.perf_counter()
    model = model_class.from_pretrained(profile.repo_id, **options).eval()
    return model, tokenizer, {"profile": profile.as_dict(), "quantization": quantization,
                              "device": device, "load_seconds": time.perf_counter() - started,
                              "torch_version": torch.__version__,
                              "transformers_version": transformers.__version__}


def save_graft(model: nn.Module, path: str | Path, profile: ModelProfile) -> None:
    """Save only new tensors. Pretrained donor tensors are not copied or overwritten."""
    from safetensors.torch import save_file
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    weights, records = {}, []
    for name, module in model.named_modules():
        if isinstance(module, QwenNRDFWrapper):
            records.append({"module": name, "config": asdict(module.config)})
            weights[name + ".gate"] = module.gate.detach().float().cpu().contiguous()
            for key, value in module.fabric.state_dict().items():
                weights[name + ".fabric." + key] = value.detach().cpu().contiguous()
    if not records:
        raise ValueError("No NRDF grafts installed")
    save_file(weights, str(directory / "graft.safetensors"))
    (directory / "graft.json").write_text(json.dumps(
        {"format": 1, "repo_id": profile.repo_id, "revision": profile.revision,
         "stage": profile.stage, "grafts": records, "validated_capability": False}, indent=2))


def load_graft(model: nn.Module, path: str | Path, profile: ModelProfile, *,
               allow_experimental: bool = False) -> None:
    from safetensors.torch import load_file
    directory = Path(path)
    metadata = json.loads((directory / "graft.json").read_text())
    if metadata.get("format") != 1 or (metadata.get("repo_id"), metadata.get("revision")) != (
            profile.repo_id, profile.revision):
        raise ValueError("Graft donor identity/revision mismatch")
    if not allow_experimental:
        raise RuntimeError("Loading an unpromoted graft requires --allow-experimental")
    tensors = load_file(str(directory / "graft.safetensors"))
    staged = []
    names = [item["module"] for item in metadata["grafts"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Missing or duplicate graft module identities")
    for entry in metadata["grafts"]:
        name = entry["module"]
        donor = model.get_submodule(name)
        if isinstance(donor, QwenNRDFWrapper):
            raise ValueError("Remove an existing graft before loading another")
        wrapper = QwenNRDFWrapper(donor, NRDFConfig(**entry["config"]))
        if wrapper.config.pulse_interval:
            from .pulse import PulseBridge
            wrapper.fabric.pulse_bridge = PulseBridge(
                model.get_input_embeddings(), model.get_output_embeddings(),
                wrapper.config.latent_dim, wrapper.hidden).to(device=wrapper.gate.device)
        prefix = name + ".fabric."
        wrapper.fabric.load_state_dict({k[len(prefix):]: v for k, v in tensors.items()
                                        if k.startswith(prefix)}, strict=True)
        with torch.no_grad():
            wrapper.gate.copy_(tensors[name + ".gate"])
        staged.append((name, wrapper))
    for name, wrapper in staged:
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, wrapper)


def prompt(model: nn.Module, tokenizer, text: str, *, max_new_tokens: int = 64,
           max_input_tokens: int = 2048) -> tuple[str, dict]:
    if not text.strip() or min(max_new_tokens, max_input_tokens) <= 0:
        raise ValueError("Nonempty prompt and positive token budgets required")
    batch = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    if batch.input_ids.shape[-1] > max_input_tokens:
        raise ValueError("Input exceeds configured limit; not silently truncating")
    device = model.get_input_embeddings().weight.device
    batch = {k: v.to(device) for k, v in batch.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    continuation = output[0, batch["input_ids"].shape[-1]:]
    return tokenizer.decode(continuation, skip_special_tokens=True), {
        "generated_tokens": len(continuation), "end_to_end_seconds": elapsed,
        "end_to_end_tokens_per_second": len(continuation) / elapsed,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "ttft_seconds": None, "inter_token_latency": None,
        "note": "Non-streaming timing includes prefill; not a decode-only throughput number"}
