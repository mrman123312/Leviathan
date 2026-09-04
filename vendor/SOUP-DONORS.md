# Additional Leviathan soup donors

This manifest records the second vendor wave: cognitive primitives, multimodal representation systems, long-context memory/inference techniques, embodied-control donors, and distributed training/evaluation infrastructure.

All entries are Git submodules pinned to exact upstream commits. Upstream licenses remain authoritative. Model checkpoints are not copied into Leviathan and may have separate or gated terms from the source repositories.

## Plasticity and concept-level cognition

| Path | Upstream | Pin | Leviathan lesson |
|---|---|---|---|
| `vendor/learning/ttt-e2e` | `test-time-training/e2e` | `a4fc4788ace38e29b5067916d4f4be33da894085` | Test-time neural adaptation with temporary learned state |
| `vendor/learning/ttt-lm` | `test-time-training/ttt-lm-pytorch` | `cd831db10c8c9a0f6340f02da5613316a8a92b67` | Learned recurrent state / neural memory |
| `vendor/representation/large-concept-model` | `facebookresearch/large_concept_model` | `fd7db8022113ae2d2ff1219d9a4cb4def897e3ff` | Sentence/concept-level latent cognition |
| `vendor/representation/sonar` | `facebookresearch/SONAR` | `3a95f405d86e2d51ba23154c8a413df34949f1c3` | Shared multilingual/multimodal concept embeddings |

## Multimodal perception and generation

| Path | Upstream | Pin | Leviathan lesson |
|---|---|---|---|
| `vendor/multimodal/bagel` | `ByteDance-Seed/Bagel` | `a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f` | Unified visual understanding/generation/editing |
| `vendor/vision/sam3` | `facebookresearch/sam3` | `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` | Persistent object-centric visual state and tracking |
| `vendor/multimodal/moshi` | `kyutai-labs/moshi` | `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` | Full-duplex streaming speech and neural audio codec state |
| `vendor/multimodal/emu3` | `baaivision/Emu3` | `dbbf9858194d70b8c58293e219ecffe22df0f9c7` | Unified discrete text/image/video prediction |
| `vendor/multimodal/janus` | `deepseek-ai/Janus` | `1daa72fa409002d40931bd7b36a9280362469ead` | Decoupled visual representations for understanding vs generation |
| `vendor/multimodal/deepseek-ocr` | `deepseek-ai/DeepSeek-OCR` | `09eaf526153e7a01ed16c9dea8c96282aaea29c0` | Optical context compression; includes OCR2 release info |

## Alternative generation algorithms

| Path | Upstream | Pin | Leviathan lesson |
|---|---|---|---|
| `vendor/architecture/dream` | `DreamLM/Dream` | `31f94a60d187e3fd481fee3bbc2c732eb94a879c` | Diffusion language generation / iterative refinement |
| `vendor/architecture/llada` | `ML-GSAI/LLaDA` | `9182493720ed723ef8031210d85959364e51cbe0` | Masked diffusion language modeling as an alternative to strict AR decoding |

## Embodied control

| Path | Upstream | Pin | Leviathan lesson |
|---|---|---|---|
| `vendor/robotics/isaac-groot` | `NVIDIA/Isaac-GR00T` | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` | Modern generalist robot/action policy stack |

## Long-context and decode efficiency

| Path | Upstream | Pin | Leviathan lesson |
|---|---|---|---|
| `vendor/inference/quest` | `mit-han-lab/Quest` | `01c1623bf9395009520874e989e29f683203b357` | Query-aware selective KV page access |
| `vendor/inference/snapkv` | `FasterDecoding/SnapKV` | `e216ddc84c5bd210378cbdbbba12ba02102aa640` | KV-cache compression and important-position retention |
| `vendor/inference/medusa` | `FasterDecoding/Medusa` | `e2a5d20c048a9b0a4092e6933c34313687422518` | Extra future-token heads / speculative decoding |

## Training and evaluation infrastructure

| Path | Upstream | Pin | Leviathan lesson |
|---|---|---|---|
| `vendor/training/megatron-lm` | `NVIDIA/Megatron-LM` | `52500b669332de180e1e04bfc8ab4cc1a2be504e` | Tensor/pipeline/expert/context parallelism for frontier-scale surgery |
| `vendor/training/torchtitan` | `pytorch/torchtitan` | `6a312901838622af315c01c4bff2a1ce43246c1e` | Clean extensible distributed training experiments |
| `vendor/evaluation/lm-evaluation-harness` | `EleutherAI/lm-evaluation-harness` | `b954108c9baaaa934b4ad842033b31a97ee30816` | Reproducible capability and regression evaluation |

## Clone

```bash
git clone --recurse-submodules https://github.com/mrman123312/Leviathan.git
```

For an existing clone:

```bash
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

The submodules contain source code only. Download model weights/checkpoints separately using each upstream project's official instructions and Leviathan's `spec/model-registry.toml` where applicable.
