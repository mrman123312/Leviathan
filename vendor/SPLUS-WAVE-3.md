# Leviathan S+ donor wave 3

This manifest records the third vendor wave added on 2026-09-04: **25 high-priority S+/S++ donors identified in the previous search plus 25 additional S+ finds**. “S+” is an internal Leviathan research-priority label, not an upstream claim.

Every source entry is pinned to a concrete upstream Git commit. Inclusion does not relicense upstream code or model weights. Model/data/assets may have terms different from the repository source license; verify each upstream license before training, redistribution, or commercial use.

## Wave A — 25 requested S+/S++ donors

| # | Path | Upstream | Pin | What Leviathan steals |
|---:|---|---|---|---|
| 1 | `vendor/memory/engram` | `deepseek-ai/Engram` | `fb7f84a21f91223715394a33a1dc24bbfb7f788e` | Conditional O(1)-style lookup memory as a second sparsity axis beside MoE |
| 2 | `vendor/precision/bitnet` | `microsoft/BitNet` | `0b341e582afbf9e1011f24744b554c96a3477eb5` | Native ternary / ~1.58-bit computation and extreme low-precision inference |
| 3 | `vendor/infrastructure/deepep` | `deepseek-ai/DeepEP` | `01dc3aaac82068020353dce2c302e38153c0bfaa` | Low-latency/high-throughput expert-parallel token dispatch and combine |
| 4 | `vendor/kernels/deepgemm` | `deepseek-ai/DeepGEMM` | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` | FP8/FP4 and grouped/Mega-MoE GEMM kernels |
| 5 | `vendor/kernels/flashmla` | `deepseek-ai/FlashMLA` | `15f13e5030374295491c5ce31b02d7e63a7772c6` | Hardware-tuned MLA prefill/decode and low-precision KV paths |
| 6 | `vendor/inference/eagle` | `SafeAILab/EAGLE` | `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` | Feature-level speculative drafting and tree verification |
| 7 | `vendor/vision/vggt` | `facebookresearch/vggt` | `a288dd0f14786c93483e45524328726ab7b1b4ce` | Explicit camera/depth/point-map/3D geometric state |
| 8 | `vendor/architecture/rwkv-lm` | `BlinkDL/RWKV-LM` | `9a75f9f037afa4418ee6283b584b92b1adb89ca1` | Attention-free/recurrent dynamic state as another memory primitive |
| 9 | `vendor/architecture/torchscale` | `microsoft/torchscale` | `4d1e0e82e5adf86dd424f1463192635b73fc8efc` | RetNet-style parallel training plus recurrent/chunkwise execution; MoE reference |
| 10 | `vendor/architecture/stripedhyena` | `togethercomputer/stripedhyena` | `7e13f618027fea9625be1f2d2d94f9a361f6bd02` | Hyena convolution + periodic attention hybrid sequence processing |
| 11 | `vendor/training/dualpipe` | `deepseek-ai/DualPipe` | `030ce4325f4ebeb437da4ebc6d00a70469dd58ae` | Bidirectional pipeline scheduling that overlaps communication with compute |
| 12 | `vendor/precision/transformer-engine` | `NVIDIA/TransformerEngine` | `ee253aa6406c84b5e75ee8b567dfb7de83e1b815` | Production FP8/FP4 mixed-precision training/inference primitives |
| 13 | `vendor/environments/browsergym` | `ServiceNow/BrowserGym` | `9e779f087de9a65668b6974d11f9ce9816026e96` | Unified browser-agent training/evaluation environments |
| 14 | `vendor/environments/osworld-v2` | `xlang-ai/OSWorld-V2` | `1c81bd34b2cbbb50d8db5b6948a956870e8e707c` | Long-horizon real-computer interaction environment |
| 15 | `vendor/data/datatrove` | `huggingface/datatrove` | `a649de79c14a550dc90f48a15c025f2dd3fd3b57` | Scalable filtering, deduplication and provenance-aware data pipelines |
| 16 | `vendor/verification/leandojo-v2` | `lean-dojo/LeanDojo-v2` | `baed5eae6e87a65a446d9f54af07aab2154e7599` | Programmatic Lean proof environment for hard formal feedback |
| 17 | `vendor/agents/ai-scientist-v2` | `SakanaAI/AI-Scientist-v2` | `96bd51617cfdbb494a9fc283af00fe090edfae48` | Hypothesis -> experiment -> analysis -> next-experiment loop |
| 18 | `vendor/data/nemo-curator` | `NVIDIA-NeMo/Curator` | `fcc800049c6e646790343cff7d4aac63fef52424` | Multimodal curation, semantic deduplication and large data processing |
| 19 | `vendor/precision/llm-awq` | `mit-han-lab/llm-awq` | `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | Activation-aware weight quantization preserving important channels |
| 20 | `vendor/precision/smoothquant` | `mit-han-lab/smoothquant` | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | Move activation outliers into weights for efficient W8A8 execution |
| 21 | `vendor/inference/recurrent-drafter` | `apple/ml-recurrent-drafter` | `bd8586bb9bbfa761644c36f932a0cd1eb3f1cdf9` | Small recurrent speculative drafter trained against a frozen large model |
| 22 | `vendor/environments/webarena-verified` | `ServiceNow/webarena-verified` | `6473f72db5dcefc97b5725b59e734504edc28a21` | More deterministic web-agent verification |
| 23 | `vendor/kernels/tilekernels` | `deepseek-ai/TileKernels` | `36d9e45d38e204ebb87e6f6e833821eee0482fe5` | Low-level tiles/operators for Engram/mHC-era architectures |
| 24 | `vendor/storage/3fs` | `deepseek-ai/3FS` | `22fca04564c7cc230fd8b9523b8b92864e1dad47` | Distributed high-throughput storage for frontier training/serving |
| 25 | `vendor/infrastructure/eplb` | `deepseek-ai/EPLB` | `d52c72d5b2f2fb4c41afbf8eb21366820239913d` | Expert placement/replication and global/hierarchical MoE load balance |

## Wave B — 25 additional S+ finds

| # | Path | Upstream | Pin | What Leviathan steals |
|---:|---|---|---|---|
| 1 | `vendor/reasoning/coconut` | `facebookresearch/coconut` | `27273cb8cca4bb763c041a63b036d0c3b7cbbb48` | Reason directly through continuous latent states instead of verbalizing every step |
| 2 | `vendor/architecture/continuous-thought-machines` | `SakanaAI/continuous-thought-machines` | `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466` | Internal temporal dynamics decoupled from input-token time |
| 3 | `vendor/inference/minference` | `microsoft/MInference` | `a4eb395f949ea39e871f9bc586d683390692c6be` | Sparse/dynamic long-context prefill and selective context computation |
| 4 | `vendor/infrastructure/mooncake` | `kvcache-ai/Mooncake` | `e58ad95342158514e273ad287e3859271f079572` | Disaggregated prefill/decode and distributed KV-cache architecture |
| 5 | `vendor/infrastructure/lmcache` | `LMCache/LMCache` | `dfc2720bb8aaf3edc6b61173018f6aad772c9369` | Reuse/move KV state across GPU, CPU, local and distributed storage |
| 6 | `vendor/inference/ktransformers` | `kvcache-ai/ktransformers` | `31985f40bcc40da08107efdb1f81bf88cb38c6b2` | CPU/GPU/NPU heterogeneous execution for giant sparse models |
| 7 | `vendor/inference/torchspec` | `lightseekorg/TorchSpec` | `130bebc92ce682f6684cc3187765938fc5a8e136` | Disaggregated draft-model/EAGLE training fed by live inference hidden states |
| 8 | `vendor/inference/tokenspeed` | `lightseekorg/tokenspeed` | `daedc96bd4fa7d6e53391cb17a3c1ba1fe658a0b` | Static compilation, C++ scheduling and frontier MoE decode optimization |
| 9 | `vendor/infrastructure/lplb` | `deepseek-ai/LPLB` | `0490f79452f7ef277e814449600b1b1dd4c663b3` | Per-batch linear-program expert load balancing |
| 10 | `vendor/training/deepspeed` | `deepspeedai/DeepSpeed` | `05daf0598b1657da832288f5e89d9e22a9a92ee5` | ZeRO/offload/distributed training and large-model systems reference |
| 11 | `vendor/training/colossalai` | `hpcaitech/ColossalAI` | `4f9953be335ef371b3848719ddafe596c01ecd37` | Alternative heterogeneous/distributed large-model training stack |
| 12 | `vendor/learning/openrlhf` | `OpenRLHF/OpenRLHF` | `3c3be6234e0cb353e76bb8019947db9dfe99fca7` | Scalable RLHF/PPO/GRPO-style post-training infrastructure |
| 13 | `vendor/learning/trl` | `huggingface/trl` | `5dd51e4d8caf495dc4cbddcddef4c659ad8174b9` | Broad post-training algorithm library and reference trainers |
| 14 | `vendor/environments/swe-smith` | `SWE-bench/SWE-smith` | `9b74ac08118a85c39c356802f7961893af73e07f` | Turn real repositories into synthetic software-engineering gyms |
| 15 | `vendor/agents/openhands` | `OpenHands/OpenHands` | `4524a919930d62535a5cdca143c8a54eaf0ede42` | Mature software-agent runtime and computer/code interaction patterns |
| 16 | `vendor/evaluation/swe-bench` | `SWE-bench/SWE-bench` | `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` | Grounded repository-level coding evaluation |
| 17 | `vendor/robotics/maniskill` | `mani-skill/ManiSkill` | `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` | High-throughput GPU robot simulation and visual manipulation tasks |
| 18 | `vendor/robotics/isaaclab` | `isaac-sim/IsaacLab` | `bffdce9d7467f349bfc8ab111fe633a0bb234851` | GPU robotics training, domain randomization and large embodied curricula |
| 19 | `vendor/robotics/mujoco` | `google-deepmind/mujoco` | `8ac5c36a2a37a6b4c17960f2615ea4df93e56241` | Clean, fast general physics environment for controlled experiments |
| 20 | `vendor/kernels/xformers` | `facebookresearch/xformers` | `9e1686f3c7a3d4af6d52bd2a03bc526cdcdb7fc0` | Optimized composable attention/Transformer operators |
| 21 | `vendor/kernels/cutlass` | `NVIDIA/cutlass` | `59e3a3338d516ca6ce0e073af8da65289678a35c` | Template-level NVIDIA GEMM/attention kernel construction |
| 22 | `vendor/kernels/triton` | `triton-lang/triton` | `1298a6b13aabbda7442ca73a83442cf7db2f608d` | Programmable GPU kernels and compiler path for custom Leviathan operations |
| 23 | `vendor/infrastructure/llm-d` | `llm-d/llm-d` | `080c14d957755da4cc363745638619fc748558f1` | Kubernetes-native distributed LLM serving, KV-aware routing and disaggregation |
| 24 | `vendor/infrastructure/nvidia-dynamo` | `ai-dynamo/dynamo` | `745b9589d6c32844168771aa565ad4704309bb5e` | Distributed inference control plane, disaggregated serving, KV-aware routing and agentic replay |
| 25 | `vendor/infrastructure/nixl` | `ai-dynamo/nixl` | `59bfaa777c1606d9d235d15852ef23a7c525bbaf` | Low-latency movement of KV/model/state data between HBM, DRAM and storage |

## Architectural interpretation

The key shift in this wave is that Leviathan now has donor implementations for **all four levels of efficiency**:

1. **Algorithmic:** latent reasoning, recurrent state, Engram memory, sparse long-context attention, speculative generation.
2. **Numerical:** ternary/FP4/FP8/W4/W8 execution.
3. **Kernel/communication:** GEMM, MLA, Triton/CUTLASS, expert all-to-all, expert placement/load balancing.
4. **Cluster/runtime:** KV reuse, disaggregated prefill/decode, storage, GPU/CPU heterogeneity, distributed routing.

It also substantially expands the grounded-learning side: Lean proof states, browser/computer environments, repository-level software gyms, physics/robotics simulators, and autonomous experiment loops.

## Clone

```bash
git clone --recurse-submodules https://github.com/mrman123312/Leviathan.git
```

For an existing checkout:

```bash
git pull
git submodule sync --recursive
git submodule update --init --recursive
```
