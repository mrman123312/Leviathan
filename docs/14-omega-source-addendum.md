# Omega model-source addendum

This file records the model-specific additions made after the original source ledger. It exists to keep three different concepts separate:

1. **base substrate** — a pretraining-stage checkpoint suitable for custom architecture/post-training work;
2. **teacher** — a strong post-trained model used to generate or critique trajectories;
3. **architecture donor** — a system whose structural ideas are worth reproducing even when its weights are not the chosen Leviathan core.

Availability, licensing and repository IDs can change. Before any large download or redistribution, re-check the current upstream model card and license.

## Qwen3-30B-A3B-Base

- Registry ID: `qwen3-30b-a3b-base`
- Upstream: `Qwen/Qwen3-30B-A3B-Base`
- Leviathan role: **experimental base substrate**
- Why: sparse MoE, small active footprint relative to total capacity, permissive release, manageable enough for repeated architecture surgery compared with trillion-parameter models.
- Primary use: Ω-S0 development.

## OLMo 3 32B Base

- Registry ID: `olmo3-32b-base`
- Upstream: `allenai/Olmo-3-1125-32B`
- Leviathan role: **scientific control**
- Why: unusually transparent training lineage and useful base checkpoint for studying when architectural changes should be introduced.

## DeepSeek-V4-Pro-Base

- Registry ID: `deepseek-v4-pro-base`
- Upstream: `deepseek-ai/DeepSeek-V4-Pro-Base`
- Leviathan role: **preferred giant semantic substrate**
- Why: true base checkpoint, very large sparse capacity, relatively low active compute versus total parameters, long-context lineage and permissive license.
- Policy: automatic weight download disabled because the checkpoint is multi-terabyte class.

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
- Not currently treated as the preferred clean Leviathan base in the Omega plan.

## GLM-5.3

- Registry ID: `glm-5.3`
- Upstream: `zai-org/GLM-5.3`
- Leviathan role: **post-training/agent teacher**
- Primary lesson: large capability gains can come from trajectory/post-training improvements even when the underlying base architecture is held relatively constant.

## Qwen3.8-2.4T-A95B

- Registry ID: `qwen3.8-2.4t-a95b`
- Upstream: `Qwen/Qwen3.8-2.4T-A95B`
- Leviathan role: **giant teacher + architecture reference**
- Primary lessons: huge sparse MoE capacity, recurrent/DeltaNet-style efficient sequence processing, periodic stronger attention and MTP.
- Important distinction: the giant public checkpoint considered here is a post-trained teacher, not Leviathan's preferred clean base substrate.

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

The teacher set is not an authority hierarchy.

Agreement among teachers increases confidence only when paired with an independent verifier appropriate to the task. Disagreement is routed into evidence gathering, execution, formal checking, experiment, or curriculum generation.

`teacher agreement != ground truth`

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
