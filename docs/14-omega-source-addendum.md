# Omega model-source addendum

This file records the model-specific additions made after the original source ledger. It exists to keep three different concepts separate:

1. **base substrate** — a pretraining-stage checkpoint suitable for custom architecture/post-training work;
2. **teacher** — a strong post-trained model used to generate or critique trajectories;
3. **architecture donor** — a system whose structural ideas are worth reproducing even when its weights are not the chosen Leviathan core.

Availability, licensing and repository IDs can change. Before any large download or redistribution, re-check the current upstream model card and license.

## DeepSeek-V4-Pro-Base

- Registry ID: `deepseek-v4-pro-base`
- Upstream: `deepseek-ai/DeepSeek-V4-Pro-Base`
- Leviathan role: **canonical pretrained semantic substrate**
- Why: true base checkpoint, very large sparse capacity, relatively low active compute versus total parameters, long-context architecture, useful expert structure for MoP decomposition and permissive license.
- Current canonical fingerprint: 61 layers, hidden size 7168, MoE intermediate size 3072, 384 routed experts, 1 shared expert, 6 routed experts/token, configured maximum positions 1,048,576 and 64 safetensors shards.
- Current MoP plan: 128-channel SwiGLU tiles -> 24 tiles/expert -> 9,216 routed tiles/layer -> 144 routed tiles/token at exact parity.
- Policy: automatic weight download disabled because the checkpoint is multi-terabyte class. `src/leviathan/deepseek_v4.py` and `spec/deepseek-v4-mop.toml` enforce the canonical full-checkpoint contract.

## Qwen3-30B-A3B-Base

- Registry ID: `qwen3-30b-a3b-base`
- Upstream: `Qwen/Qwen3-30B-A3B-Base`
- Leviathan role: **development/regression control**
- Why: sparse MoE, small active footprint relative to total capacity, permissive release and manageable enough for repeated architecture debugging compared with trillion-parameter models.
- Primary use: cheap controls, interface tests and training-loop debugging.
- Important distinction: success on Qwen does not count as a canonical full-V4 Leviathan result.

## OLMo 3 32B Base

- Registry ID: `olmo3-32b-base`
- Upstream: `allenai/Olmo-3-1125-32B`
- Leviathan role: **scientific control**
- Why: unusually transparent training lineage and useful base checkpoint for studying when architectural changes should be introduced.

## MiMo-V2.5-Pro-Base

- Registry ID: `mimo-v2.5-pro-base`
- Upstream: `XiaomiMiMo/MiMo-V2.5-Pro-Base`
- Leviathan role: **frontier efficiency substrate / donor**
- Primary lessons: local/global attention economy, sparse MoE, MTP and reduced KV-cache pressure.
- Policy: frontier download disabled by default.

## Mistral Large 3 Base

- Registry ID: `mistral-large-3-base`
- Upstream: `mistralai/Mistral-Large-3-675B-Base-2512`
- Leviathan role: **multimodal base donor**
- Primary lesson: inherit a clean multimodal representation stack and connect it through learned projection bridges rather than incompatible raw weight merging.

## GLM-5.3-Flash

- Registry ID: `glm-5.3-flash`
- Upstream: `zai-org/GLM-5.3-Flash`
- Leviathan role: **efficiency teacher + architecture donor**
- Primary lessons: low active compute, hybrid sparse/linear sequence processing, richer residual connectivity and multimodal efficiency.
- Not treated as the canonical clean Leviathan base.

## GLM-5.3

- Registry ID: `glm-5.3`
- Upstream: `zai-org/GLM-5.3`
- Leviathan role: **post-training/agent teacher**
- Primary lesson: large capability gains can come from trajectory/post-training improvements even when the underlying base architecture is held relatively constant.

## Qwen3.8-2.4T-A95B

- Registry ID: `qwen3.8-2.4t-a95b`
- Upstream: `Qwen/Qwen3.8-2.4T-A95B`
- Leviathan role: **giant teacher + architecture reference**
- Primary lessons: huge sparse MoE capacity, efficient sequence processing and MTP.
- Important distinction: the giant public checkpoint considered here is a post-trained teacher, not Leviathan's chosen clean pretrained core.

## DeepSeek-V4-Pro-0813

- Registry ID: `deepseek-v4-pro-0813`
- Upstream: `deepseek-ai/DeepSeek-V4-Pro-0813`
- Leviathan role: **behavior teacher** paired conceptually with the V4 Pro Base substrate.
- Use: compare what post-training added over the base representation and distill verified trajectory improvements selectively.

## Kimi K3

- Registry ID: `kimi-k3`
- Registry placeholder upstream: `MoonshotAI/Kimi-K3`
- Leviathan role: **teacher + architecture donor**
- Primary lessons: KDA, Attention Residuals, latent MoE, quantization-native design, persistent long-horizon agent state and vision-in-the-loop correction.
- Important: exact checkpoint/repository naming and current license must be verified before automation; therefore automatic download is disabled.

## Qwen3-Omni

- Registry ID: `qwen3-omni`
- Registry placeholder upstream: `Qwen/Qwen3-Omni`
- Leviathan role: **multimodal-output architecture donor**
- Primary lesson: separate semantic cognition from modality-specific realization such as streaming speech.
- Exact checkpoint selection is intentionally left experiment-specific.

## Teacher-ensemble rule

The teacher set is not an authority hierarchy and it is not the deployed cognitive model.

Agreement among teachers increases confidence only when paired with an independent verifier appropriate to the task. Disagreement is routed into evidence gathering, execution, formal checking, experiment, or curriculum generation.

`teacher agreement != ground truth`

The single-cognitive-model invariant refers to the canonical Leviathan semantic path. Offline teachers, deterministic tools and independent verifiers may supply training/evaluation evidence without becoming hidden runtime foundation-model branches.

## Provenance requirement

Any experiment that uses one of these models must record:

- registry ID;
- exact upstream repo ID;
- immutable upstream revision;
- model config hash;
- tokenizer revision;
- license checked date;
- quantization/precision;
- serving/training code revision;
- hardware topology;
- whether the model acted as substrate, donor, teacher, verifier, or control.

For canonical DeepSeek V4 experiments also record:

- full 64-shard verification result;
- MoP tile width;
- active tiles/token;
- router revision;
- parity evidence against the frozen pretrained baseline;
- protected benchmark results;
- real wall-clock efficiency metrics.
