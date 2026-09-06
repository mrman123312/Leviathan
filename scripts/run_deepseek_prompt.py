#!/usr/bin/env python3
"""Run prompts against Leviathan's canonical DeepSeek V4 substrate.

Two backends are supported:

1. endpoint: talk to an already-served OpenAI-compatible V4 instance. This is the
   practical high-throughput path and requires no local PyTorch installation.
2. transformers: load a local checkpoint directly. With --mop0-reference, routed
   experts are replaced by Leviathan's deliberately slow exact-tile reference path.

DeepSeek-V4-Pro-Base is a base/pretrained model, so raw /v1/completions is the default.
Use --chat only when the served/local checkpoint has an appropriate chat template.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leviathan.deepseek_v4 import CANONICAL_REPO_ID  # noqa: E402
from leviathan.inference_client import (  # noqa: E402
    ChatRequest,
    CompletionRequest,
    chat,
    complete,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("endpoint", "transformers"),
        default="endpoint",
        help="Use an OpenAI-compatible server or load the local checkpoint directly.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. If omitted, enter an interactive prompt loop.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LEVIATHAN_SERVED_MODEL", CANONICAL_REPO_ID),
        help="Model name sent to the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LEVIATHAN_INFERENCE_URL", "http://127.0.0.1:8000"),
        help="OpenAI-compatible server base URL for --backend endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LEVIATHAN_INFERENCE_API_KEY"),
        help="Optional bearer token for the inference server.",
    )
    parser.add_argument("--chat", action="store_true", help="Use chat-completions/chat template.")
    parser.add_argument("--system", default=None, help="Optional system message in --chat mode.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)

    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "LEVIATHAN_DEEPSEEK_V4_DIR",
            "models/checkpoints/deepseek-v4-pro-base",
        ),
        help="Local checkpoint path for --backend transformers.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. The full V4 model normally requires distributed hardware.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow checkpoint-provided Transformers code when required.",
    )
    parser.add_argument(
        "--mop0-reference",
        action="store_true",
        help="Replace routed experts with the exact but very slow 128-channel MoP-0 reference path.",
    )
    parser.add_argument("--tile-width", type=int, default=128)
    return parser.parse_args()


def endpoint_once(args: argparse.Namespace, prompt: str) -> str:
    if args.chat:
        messages: list[dict[str, str]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        text, _ = chat(
            args.base_url,
            ChatRequest(
                model=args.model,
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ),
            api_key=args.api_key,
            timeout=args.timeout,
        )
        return text

    text, _ = complete(
        args.base_url,
        CompletionRequest(
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
        api_key=args.api_key,
        timeout=args.timeout,
    )
    return text


def _input_device(model: Any) -> Any:
    try:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        if str(device) != "meta":
            return device
    except Exception:
        pass
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RuntimeError("could not determine an input device for the loaded model")


def load_local(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Local inference dependencies are missing. Install them with: "
            "python -m pip install -e '.[inference]'"
        ) from exc

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "dtype": "auto",
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.mop0_reference:
        # Keep the donor baseline and our Python tile wrapper on the same explicit
        # expert implementation. Grouped/deepgemm kernels are the later speed path;
        # the reference path must be inspectable and not silently bypassed.
        model_kwargs["experts_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    model.eval()

    if args.mop0_reference:
        from leviathan.mop0_reference import install_mop0_reference

        report = install_mop0_reference(model, tile_width=args.tile_width)
        print(
            "MoP-0 reference installed: "
            f"{report.wrapped_experts} routed experts across "
            f"{report.moe_modules} MoE modules. This path is a parity oracle, not a speed path.",
            file=sys.stderr,
        )
    return tokenizer, model


def local_once(args: argparse.Namespace, tokenizer: Any, model: Any, prompt: str) -> str:
    import torch

    if args.chat:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("tokenizer does not provide a chat template")
        messages: list[dict[str, str]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt")

    device = _input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = int(encoded["input_ids"].shape[-1])

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        output = model.generate(**encoded, **generation_kwargs)
    new_tokens = output[0, input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def prompt_loop(args: argparse.Namespace) -> int:
    tokenizer = model = None
    if args.backend == "transformers":
        tokenizer, model = load_local(args)

    def run_one(prompt: str) -> str:
        if args.backend == "endpoint":
            return endpoint_once(args, prompt)
        return local_once(args, tokenizer, model, prompt)

    if args.prompt is not None:
        print(run_one(args.prompt))
        return 0

    print(
        f"Leviathan prompt shell | backend={args.backend} | "
        f"mop0_reference={args.mop0_reference}\n"
        "Type /exit to quit."
    )
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.strip() in {"/exit", "/quit"}:
            return 0
        if not prompt.strip():
            continue
        try:
            print(run_one(prompt))
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
    return 0


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.tile_width <= 0:
        raise SystemExit("--tile-width must be positive")
    if args.mop0_reference and args.backend != "transformers":
        raise SystemExit("--mop0-reference requires --backend transformers")
    return prompt_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
