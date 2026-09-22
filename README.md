# High-Performance GPT-2 Pre-Training Engine from Scratch

A production-grade, system-optimized PyTorch implementation of the **GPT-2 (124M)** architecture trained from scratch on the **FineWeb-EDU** dataset. 

This repository focuses on low-level deep learning systems engineering, demonstrating high-throughput pre-training pipelines, Distributed Data Parallel (DDP) scaling, PyTorch 2.0+ kernel optimizations, and custom benchmark evaluations.

---

## Key Technical Features

* **PyTorch 2.0+ Kernel Fusion:** Native integration of `torch.compile` and `F.scaled_dot_attention` (FlashAttention) for maximal GPU FLOP utilization.
* **Distributed Multi-GPU Scaling:** Fully integrated PyTorch Distributed Data Parallel (DDP) with manual gradient synchronization control (`require_backward_grad_sync`) during gradient accumulation steps.
* **Mixed Precision & Memory Optimization:** Automatic mixed-precision (`bfloat16` / `float16` autocast) combined with vocabulary padding to $50,304$ (a multiple of 64) for tensor-core memory alignment.
* **Fused Optimizer & Schedulers:** Fused `AdamW` optimizer with decoupling of weight decay (2D weight matrices decayed, 1D biases/LayerNorms non-decayed) and a cosine learning rate decay schedule with linear warmup.
* **Uncompiled Eager Mode Unwrapping:** Dynamic sequence length handling during HellaSwag zero-shot evaluation and top-$k$ text generation by bypassing `torch.compile` via `raw_model` unwrapping.
* **Data Streaming Pipeline:** Custom `Dataloaderlite` for streaming tokenized FineWeb-EDU dataset shards (`.npy`) across multiple distributed worker processes without redundant memory overhead.

---

## Model Architecture Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | 124M | Standard GPT-2 Small configuration |
| **Layers (`n_layer`)** | 12 | Transformer block depth |
| **Attention Heads (`n_head`)** | 12 | Multi-head self-attention heads |
| **Embedding Dim (`n_embd`)** | 768 | Model hidden dimension |
| **Context Length (`block_size`)** | 1024 | Maximum token sequence length |
| **Vocabulary Size** | 50,304 | Padded GPT-2 Tiktoken BPE vocab |

---

## Repository Structure

```text
├── train.py           # Master training script (DDP, compile, eval, train loop)
├── hellaswag.py       # HellaSwag evaluation parsing and rendering helpers
├── edu_fineweb10B/    # Directory containing tokenized FineWeb-EDU shard files (.npy)
├── logs/              # Log output directory (contains train/val logs and checkpoints)
└── README.md          # Project documentation

Quickstart & Usage
1. Requirements & Setup
Install the required dependencies:

Bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install tiktoken transformers datasets numpy
Download the HellaSwag evaluation module:

Bash
wget [https://raw.githubusercontent.com/karpathy/build-nanogpt/master/hellaswag.py](https://raw.githubusercontent.com/karpathy/build-nanogpt/master/hellaswag.py)
2. Dataset Setup
Place your tokenized FineWeb-EDU numpy shards inside the edu_fineweb10B/ directory:

Bash
# Example directory format:
edu_fineweb10B/
├── edufineweb_train_000001.npy
├── edufineweb_val_000001.npy
3. Launching Training
Single-GPU / Google Colab Execution:
Bash
python train.py
Multi-GPU Execution (Distributed Data Parallel):
Bash
torchrun --nproc_per_node=4 train.py
Convergence & Verification Run
The training pipeline was validated via short verification runs on FineWeb-EDU to confirm system stability and mathematical convergence.

Theoretical Baseline vs. Empirical Loss
For a vocabulary size of V=50,304, equal probability guessing yields an initial cross-entropy loss of:

−ln( 
50304
1
​
 )≈10.825
Sample Terminal Output
Plaintext
total desired batch size : 65536
=> calculated gradient accumulation steps : 4
found 1 shards for split train
found 1 shards for split val
using fused AdamW: True
validation loss: 10.8214
step    0 | loss: 10.822117 | lr: 1.8000e-04 | norm: 0.7889 | dt: 3120.45ms | tok/sec: 21002.11
step   10 | loss: 10.790219 | lr: 1.8000e-03 | norm: 0.5567 | dt: 2850.12ms | tok/sec: 22994.18
step   20 | loss: 10.625011 | lr: 1.5628e-03 | norm: 1.2285 | dt: 2845.82ms | tok/sec: 23028.89
step   30 | loss: 10.338568 | lr: 9.9000e-04 | norm: 0.7555 | dt: 2818.44ms | tok/sec: 23252.68
step   35 | loss: 10.182283 | lr: 6.8003e-04 | norm: 0.3791 | dt: 2857.33ms | tok/sec: 22936.44
Evaluation & Benchmarks
The training engine integrates two built-in evaluation loops:

Validation Loss Tracking: Computes cross-entropy loss over evaluation shards periodically to monitor overfitting.

HellaSwag Zero-Shot Evaluation: Evaluates commonsense reasoning by scoring token completion probabilities across four candidate sentences using length-normalized accuracy (acc_norm).

Autoregressive Text Generation: Executes top-k (k=50) multinomial sampling to generate sample text completions at regular step intervals.

To prevent PyTorch's compiler from constantly re-compiling execution graphs due to dynamic sequence lengths in HellaSwag and text generation, all evaluation calls route directly through the uncompiled raw_model:

Python
# Extract uncompiled model for dynamic generation/evaluation
raw_model = model.module if ddp else (model._orig_mod if hasattr(model, '_orig_mod') else model)

# Forward evaluation through raw_model eager mode
with torch.no_grad():
    logits = raw_model(tokens)
Compute Constraint Notice
This repository focuses on low-level systems engineering, architecture design, and training loop optimization. Full 10B-token training runs require dedicated multi-node GPU clusters; verification checkpoints and logs provided here reflect smoke-test validation runs.
