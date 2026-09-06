# Recurrent-depth research audit

Primary sources informed the design; NRDF hypotheses are not findings attributed to those papers. No original large training run is claimed reproduced.

| Source | Implementation lesson | Qualification |
|---|---|---|
| Universal Transformers, 1807.03819 | Shared transformations and adaptive per-position compute | A threshold is not a trained halting policy |
| Scaling up Test-Time Compute, 2502.05171v2 | Prelude/core/coda, repeated input injection, variable depth, normalization | Looping arbitrary pretrained blocks is not equivalent; step conditioning can harm path independence in their ablations |
| Ouro, 2510.25741 | Looped training and depth allocation | Requires training, not simply changing an inference loop count |
| LoopFormer, 2602.11451v1 | Time/step conditioning and shortcut consistency | Contrasts with a universal no-step-conditioning rule; keep a controlled ablation |
| Loop, Think, & Generalize, 2604.07822 | Extrapolation and overthinking tests | Controlled task results do not guarantee Qwen gains |
| Mechanistic Analysis, 2604.11791 | Input injection, normalization, cycles/fixed points | Latent convergence is not a correctness certificate |
| Universal Transformers Need Memory, 2604.21999 | Depth/state trade-off and early-halting failure | Does not establish universal memory dimensions |
| LOTUS, 2606.31779 | Parallel latent slots with explicit-step supervision | Random token-pulse bridges do not inherit useful latent CoT |

Primary papers:

- https://arxiv.org/abs/1807.03819
- https://arxiv.org/html/2502.05171v2
- https://arxiv.org/abs/2510.25741
- https://arxiv.org/html/2602.11451v1
- https://arxiv.org/abs/2604.07822
- https://arxiv.org/abs/2604.11791
- https://arxiv.org/abs/2604.21999
- https://arxiv.org/html/2606.31779v1

Implementation inspection:

- https://github.com/seal-rg/recurrent-pretraining : README and `recpre/raven_modeling_minimal.py`, including prelude/core/coda and cache interfaces.
- https://github.com/armenjeddi/loopformer : official repository identity/README and the paper's concrete PyTorch pseudocode; its full training run was not reproduced.

Model sources:

- https://huggingface.co/Qwen/Qwen3.8-27B : explicitly post-trained, not base.
- https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json : actual Qwen3_5ForConditionalGeneration configuration.
- https://huggingface.co/Qwen/Qwen3-1.7B-Base : explicitly pretraining-only.
- https://huggingface.co/Qwen/Qwen3-1.7B-Base/raw/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/config.json : Qwen3ForCausalLM, hidden2048, FFN6144, 28 layers.
- https://huggingface.co/docs/transformers/quantization/bitsandbytes : NF4 loading API.

MTP metadata does not prove a given runtime exposes usable speculative heads. No secret OpenAI recurrent architecture is assumed. Weight sharing reduces parameter duplication, not automatically FLOPs, latency or energy. Four-bit quantization is approximate, not universally lossless.
