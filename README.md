---
license: apache-2.0
base_model: mlx-community/Qwen3.6-27B-4bit
tags:
- mlx
- lora
- fine-tuned
- vram-optimized
- 36gb-mac
---

# Qwen 3.6 27B Fine-Tuned (MLX Fused / 4-bit)

This repository contains the fine-tuned, merged version of [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit). The training was carried out locally on a 36 GB Apple Silicon Mac using advanced memory optimization techniques.

## Training Optimizations for Strict VRAM Limits (36 GB Mac)
Fine-tuning a 27B parameter model typically requires enterprise-grade GPUs. To fit this training successfully inside a 36 GB unified memory footprint, we developed several custom configurations and monkeypatches:

1. **Micro-Batch Size Reduction**: Capped `per_device_train_batch_size = 1` to reduce active batch activation memory.
2. **Scaled Gradient Accumulation**: Set `gradient_accumulation_steps = 36` to maintain an effective update batch size of 36 without memory bloat.
3. **LoRA Rank and Noise Reduction**: Lowered LoRA rank `r` to `8` and set `lora_dropout = 0.0` to minimize trainable parameters and remove numeric noise.
4. **Context Length Truncation**: Capped context lengths to `512` tokens, decreasing attention matrix memory requirements by 16x.
5. **Partial Layer Fine-Tuning**: Froze the first 48 layers of the 64-layer Qwen architecture, targeting only the last 16 layers (`num_layers = 16`). This cut backpropagation gradient tracking by 75%.
6. **Explicit Gradient Checkpointing**: Recomputed activations during the backward pass rather than caching them, saving massive activation memory.
7. **Aggressive Cache Purging**: Monkeypatched SFTTrainer to force `clear_cache_threshold = 0`, forcing MLX to release its GPU allocator cache after every single training step.
8. **Differentiable Model Switch**: Switched from mixed-precision GLA models (which contain custom kernels lacking registered Vector-Jacobian Product gradients) to a standard quantized 4-bit model that uses fully-differentiable `QuantizedLinear` layers.
9. **Validation Memory Conservation**: Passed a single-sample validation set (`dataset.select([0])`) during training to prevent MLX from tokenizing and caching the entire 12,181 evaluation set in RAM.
10. **Autodiff-Safe Masked Loss (Anti-NaN)**: Capped prompts could truncate the assistant's response entirely, resulting in 0 target tokens. Standard conditional masking leads to `nan * 0` in autodiff backward passes, which corrupts gradients. We patched the loss function with an autodiff-safe mask that selects a dummy token when target length is 0, outputting clean, zero gradients.
11. **Gradient Clipping**: Injected L2 norm gradient clipping (threshold of `1.0`) into the optimizer's update step to stabilize parameter updates and prevent gradient explosion.

## Hyperparameters

- **Base Model**: `mlx-community/Qwen3.6-27B-4bit`
- **Optimizer**: AdamW
- **Learning Rate**: `3e-5` (Linear decay to 0.0)
- **Epochs/Steps**: 350 steps (~1 epoch)
- **Prompt Masking**: Response-only masking (`mask_prompt=True`)

## How to use in MLX

```python
from mlx_lm import load, generate

model, tokenizer = load("your-username/Qwen3.6-27B-Finetuned-MLX")

response = generate(
    model, 
    tokenizer, 
    prompt="<|im_start|>user\nWhat is the oldest tree?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    verbose=True
)
```
