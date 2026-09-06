# Engineering steering ledger

| Evidence/failure | Rule |
|---|---|
| Prior small sparse paths were slower | Require workload-specific real wall-clock measurements |
| In-place post-recruitment updates broke autograd | Revise out-of-place; commit after discussion |
| Zero-gate branches can block learning | Keep the training graph; exact bypass is inference-only |
| Global cell state mixes unrelated requests | Explicit token-local depth state; no global batch averaging |
| Packed weights are not ordinary matrices | Format-aware accessors or fail closed |
| Official Qwen27B is post-trained | Label it honestly; separately pin a genuine base |
| Tiny FFN cache slower, larger FFN cache faster | Admit cache optimization per workload, not globally |
| Synthetic depth6 worse than depth2 | No more-depth-is-better claim; evaluate depth/cost curves |
| Confidence and variance are uncalibrated | Consensus is not truth; no abstention authority without evidence |
| Full-prefix self-speculation is a correctness reference | No throughput claim without cache-aware GPU measurements |
| Actual 1.7B zero-graft logits unchanged; ARC smoke 4/8 before and after | Preservation evidence only, not reasoning improvement |
| Buffer-only INT4 modules have no Parameters | Infer device from parameters or buffers, and test both |

Append new counterexamples and evidence. This repository ledger is not a promise of persistent assistant memory.
