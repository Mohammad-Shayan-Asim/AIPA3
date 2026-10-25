# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Assignment

AI624 "AI on Edge Devices" — Assignment 3: Quantization (due 2026-10-23 11:59 PM).

- **Task 1** (20 pts): GPT-2 baseline performance and profiling on WikiText-2
- **Task 2** (60 pts): GPT-2 Post-Training Quantization — Linear PTQ, SmoothQuant, GPTQ
- **Task 3** (20 pts): ResNet-18 Quantization-Aware Training on CIFAR-10

Deliverable: one ZIP containing a main Jupyter notebook (with all outputs visible), any supporting Python modules, and a Markdown report.

## Environment

```bash
source venv/bin/activate   # Python 3.12
pip install <package>      # install into the venv
```

Key dependencies: `torch`, `torchvision`, `transformers`, `datasets`, `torchprofile`, `calflops`, `tqdm`.

## Module Overview

| File | Purpose |
|------|---------|
| `resnet.py` | ResNet18/34/50 adapted for CIFAR-10 (3×3 stem, stride 1 — **not** ImageNet's 7×7/stride-2 stem). Pretrained weights load from `state_dicts/<arch>.pt`. |
| `data.py` | `CIFAR10Data` — wraps train/val/test `DataLoader`s with standard CIFAR-10 normalization. |
| `utils.py` | `train_model()` train loop, `preds()` batch inference, `compress()` gzip-serialize a state dict and return its byte size. |
| `profiler.py` | `profile()` — GPU-only; measures median latency, peak/avg memory, power draw (via `torch.cuda.power_draw`), MACs/FLOPs via `calflops`, and energy per forward pass. Requires CUDA. |

## Key Implementation Details

**GPT-2 projection modules use `Conv1D`, not `nn.Linear`.**  
The weight matrix is transposed relative to `nn.Linear` (`weight.shape = [in, out]` vs `[out, in]`). Verify axes before implementing quantization or replacing modules.

**Quantization scope (Task 2):** quantize weights *and* input activations of these four projection types in all 12 blocks:
- `transformer.h[i].attn.c_attn` (Q/K/V)
- `transformer.h[i].attn.c_proj` (attention output)
- `transformer.h[i].mlp.c_fc` (MLP input)
- `transformer.h[i].mlp.c_proj` (MLP output)

Keep biases, embeddings, tied vocab head, LayerNorm, GELU, softmax, residual additions, and attention-score/AV multiplications in FP32. Do **not** quantize the KV cache.

**Quantization conventions:**
- Weights: symmetric per-output-channel, `S_W^(c) = max_i|W_ic| / Q_b`, zero-point = 0
- Activations: asymmetric per-tensor, unsigned INT8 range [0, 255]
- W8A8 and W4A8 configs; Static (fixed calibration range) and Dynamic (runtime range) variants

**WikiText-2 setup:**
- Dataset: `wikitext-2-raw-v1` from `Salesforce/wikitext`; sequence length fixed at 128 tokens
- Calibration: 128 sequences from the training split (record and reuse seed)
- Tokenization: concatenate non-empty examples with end-of-text separator, form non-overlapping 128-token blocks, score 127 next-token targets per block (skip position 0)
- Use the **test split** only for final evaluation; tune hyperparameters on validation split
- Run inference with `use_cache=False`

**Profiling requirements (Task 1 & 2):** 10 warm-up + 30 measured forward passes, GPU synchronized, batch size 1 for profiling. Report: parameter count, serialized size (MB), MACs/FLOPs, mean±std latency, sequences/sec, tokens/sec, peak GPU memory, power (W), energy (mJ) per forward pass.

**Implementation constraint:** do not use SmoothQuant, GPTQ, or fake-quantization functions from external packages. Implement quantization logic from scratch; standard model loading, dataset utilities, and profiling tools are allowed.
