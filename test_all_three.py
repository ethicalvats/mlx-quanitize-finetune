import sys
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
import gc

prompt = "<|im_start|>user\nWhat is the oldest tree?<|im_end|>\n<|im_start|>assistant\n<think>\n"
sampler = make_sampler(temp=0.0)

print("--- 1. GENERATING WITH BASE MODEL ONLY ---")
model_base, tokenizer_base = load("mlx-community/Qwen3.6-27B-4bit")
out_base = generate(
    model_base,
    tokenizer_base,
    prompt=prompt,
    max_tokens=60,
    sampler=sampler,
    verbose=True
)

# Free VRAM
del model_base
del tokenizer_base
gc.collect()
mx.clear_cache()

print("\n--- 2. GENERATING WITH BASE MODEL + ADAPTERS ---")
model_adapter, tokenizer_adapter = load(
    "mlx-community/Qwen3.6-27B-4bit",
    adapter_path="lora_finetuned/adapters"
)
out_adapter = generate(
    model_adapter,
    tokenizer_adapter,
    prompt=prompt,
    max_tokens=60,
    sampler=sampler,
    verbose=True
)

# Free VRAM
del model_adapter
del tokenizer_adapter
gc.collect()
mx.clear_cache()

print("\n--- 3. GENERATING WITH FUSED MODEL ---")
model_fused, tokenizer_fused = load(
    "./Qwen3.6-27B-Claude-Opus-Reasoning-Distilled-MLX"
)
out_fused = generate(
    model_fused,
    tokenizer_fused,
    prompt=prompt,
    max_tokens=60,
    sampler=sampler,
    verbose=True
)
