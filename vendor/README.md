# Vendored upstream research stack

Leviathan vendors public upstream source projects as **pinned Git submodules**. A submodule is a real clone of the upstream repository at a specific commit, while preserving the upstream project's own history, authorship and license.

Clone everything:

```bash
git clone --recurse-submodules https://github.com/mrman123312/Leviathan.git
cd Leviathan
git submodule update --init --recursive
```

If Leviathan was already cloned:

```bash
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

## Rules

- Every `vendor/` gitlink is pinned to a concrete upstream commit for reproducibility.
- Each upstream project's own license remains authoritative. Inclusion here does **not** relicense upstream code.
- `open-weight` or `source-available` projects are included when their GitHub source repository is publicly cloneable, but are labeled as such in Leviathan's source ledger.
- Large Hugging Face / ModelScope checkpoints are not copied into this repository. Their source repos are vendored and checkpoint locations remain documented separately.
- Closed systems (GPT Astra/Sol/Luna, Fable, AlphaEvolve runtime, SIMA 2, Genie 3) are **not** vendored. Leviathan contains only the public lessons/references learned from them.
- Dolphin 3.0 R1 Mistral 24B was discussed as a Hugging Face checkpoint; no dedicated official GitHub source repository was found, so it is not represented as a submodule.

## Pinned manifest

### World models

| Path | Upstream | Pin |
|---|---|---|
| `vendor/world-models/nvidia-cosmos` | `NVIDIA/Cosmos` | `5b50bba0fb97f4aa07b03d299bc1a22bafa10ade` |
| `vendor/world-models/cosmos-predict2.5` | `nvidia-cosmos/cosmos-predict2.5` | `a2c298b0a3df3778b973fe65e9e58877b292d8a7` |
| `vendor/world-models/vjepa2` | `facebookresearch/vjepa2` | `204698b45b3712590f06245fbfba32d3be539812` |
| `vendor/world-models/jepa-wms` | `facebookresearch/jepa-wms` | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |

### Reasoning / foundation / sparse compute

| Path | Upstream | Pin |
|---|---|---|
| `vendor/reasoning/deepseek-r1` | `deepseek-ai/DeepSeek-R1` | `0cf78561f1d51c84a21b2190626b21116d5c68bb` |
| `vendor/reasoning/qwen3` | `QwenLM/Qwen3` | `7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e` |
| `vendor/reasoning/kimi-k3` | `MoonshotAI/Kimi-K3` | `3cb39dfd32e51c3328e2e4b4af21341247d06c43` |
| `vendor/reasoning/step-3.5-flash` | `stepfun-ai/Step-3.5-Flash` | `21d85a5f6c291f3f138da0bc09979af43345251a` |
| `vendor/reasoning/longcat-flash-thinking` | `meituan-longcat/LongCat-Flash-Thinking` | `ea315cc69cc098182494e789d84099f272354463` |
| `vendor/reasoning/longcat-flash-chat` | `meituan-longcat/LongCat-Flash-Chat` | `e929faeb26fab3eb8f6e0e294e94b7be1e785d4f` |
| `vendor/reasoning/k2-horizon-post-train` | `ifm-ai/horizon-post-train` | `023c0f7b9de5da693d435a196b0fd436dddb0438` |
| `vendor/reasoning/minimax-m2.5` | `MiniMax-AI/MiniMax-M2.5` | `0fe00c843c16e7081a9631daeafc11288f5f871c` |

### Reinforcement learning / self-generated experience

| Path | Upstream | Pin |
|---|---|---|
| `vendor/learning/verl` | `verl-project/verl` | `3d36367e83d7130f0b658b6be46db96d85ff9ce7` |
| `vendor/learning/absolute-zero-reasoner` | `LeapLabTHU/Absolute-Zero-Reasoner` | `484afa480c8f6fd77faa3d35451f24f287f58ee1` |

### Persistent memory

| Path | Upstream | Pin |
|---|---|---|
| `vendor/memory/letta` | `letta-ai/letta` | `4511fa0bc91f68fbab32b91f694617271ea9012b` |
| `vendor/memory/mem0` | `mem0ai/mem0` | `9a7924befd7026e41e445ba809370009e5e985a6` |

### Verification / reward / process supervision

| Path | Upstream | Pin |
|---|---|---|
| `vendor/verification/deepseek-prover-v2` | `deepseek-ai/DeepSeek-Prover-V2` | `e598a57ea3284997d4a2a168a069fdd5064afbc8` |
| `vendor/verification/skywork-reward` | `SkyworkAI/Skywork-Reward` | `03c205fce84ad2ee6cc9c40414d5bca5bd79e84a` |
| `vendor/verification/evpv-prm` | `Qwen-Applications/EVPV-PRM` | `3fadeec191b15a4dc66ac8935515ed83d064820c` |

### Robotics / native action

| Path | Upstream | Pin |
|---|---|---|
| `vendor/robotics/openpi` | `Physical-Intelligence/openpi` | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| `vendor/robotics/lerobot` | `huggingface/lerobot` | `3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e` |
| `vendor/robotics/rdt-1b` | `thu-ml/RoboticsDiffusionTransformer` | `cd79363a1387e8f81c7724d070ef7e45fd23150f` |
| `vendor/robotics/octo` | `octo-models/octo` | `241fb3514b7c40957a86d869fecb7c7fc353f540` |

### Alternative sequence architectures / representation

| Path | Upstream | Pin |
|---|---|---|
| `vendor/architecture/blt` | `facebookresearch/blt` | `9774ed4fcc78313f9f218295f3d7e4decdadf2ae` |
| `vendor/architecture/mamba` | `state-spaces/mamba` | `e9594ce1c732d97440f0332fdc43170a2294dbfa` |
| `vendor/architecture/recurrentgemma` | `google-deepmind/recurrentgemma` | `2efa84dac0e68e63547a27a18fa943c98f1c312e` |

### Native multimodality

| Path | Upstream | Pin |
|---|---|---|
| `vendor/multimodal/qwen3-omni` | `QwenLM/Qwen3-Omni` | `e4235853125589c789f06a2dd83e9f4126df5e9d` |

### GUI perception / actions

| Path | Upstream | Pin |
|---|---|---|
| `vendor/gui/gui-actor` | `microsoft/GUI-Actor` | `d98d1bbd01862f9112114b83b032f492c365a173` |
| `vendor/gui/showui` | `showlab/ShowUI` | `21ed7cb24be0cc877bb8352ee34d58a9aea2c876` |

### Agent architectures / tool use

| Path | Upstream | Pin |
|---|---|---|
| `vendor/agents/voyager` | `MineDojo/Voyager` | `55e45a880755d0c8c66ca7fb5fe7962ac8974f89` |
| `vendor/agents/swe-agent` | `SWE-agent/SWE-agent` | `3ea751c087f32b16e039a2233dd6eefecef325d5` |
| `vendor/agents/gorilla` | `ShishirPatil/gorilla` | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` |
| `vendor/agents/ufo` | `microsoft/UFO` | `364eb7969d392e857299ceaf14bd6057e5b00078` |

### Inference / serving / kernels

| Path | Upstream | Pin |
|---|---|---|
| `vendor/inference/vllm` | `vllm-project/vllm` | `8a728663c1c3eeace834a95f5654fa653cc1998c` |
| `vendor/inference/sglang` | `sgl-project/sglang` | `e3305b3b87e66cf4cdcf7ab8c0ed5dd8d1f29591` |
| `vendor/inference/flash-attention` | `Dao-AILab/flash-attention` | `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` |
| `vendor/inference/flashinfer` | `flashinfer-ai/flashinfer` | `60b49158ab4fb81718aef486c2d3c89aec4c1901` |
| `vendor/inference/tensorrt-llm` | `NVIDIA/TensorRT-LLM` | `cf45d3ce8bcbb3b365f329ef3be1e5ff9c112efe` |

## Updating pins

Pins should be updated intentionally, after reviewing upstream changes:

```bash
git submodule update --remote vendor/<path>
git add .gitmodules vendor/<path>
git commit -m "Update <project> vendor pin"
```

Do not blindly track every upstream `main`: reproducible experiments require known source revisions.
