import sys
from mlx_lm import load, generate

prompt = "<|im_start|>user\nWhat is the oldest tree?<|im_end|>\n<|im_start|>assistant\n<think>\n"

print("--- LOADING BASE MODEL + ADAPTERS ---")
model_adapter, tokenizer_adapter = load(
    "mlx-community/Qwen3.6-27B-4bit",
    adapter_path="lora_finetuned/adapters"
)

print("\n--- GENERATING WITH ADAPTERS ---")
out_adapter = generate(
    model_adapter,
    tokenizer_adapter,
    prompt=prompt,
    max_tokens=100,
    verbose=True
)

# Force delete to free VRAM before loading the fused model
del model_adapter
del tokenizer_adapter
import gc
import mlx.core as mx
mx.metal.clear_cache()
gc.collect()

print("\n\n--- LOADING FUSED MODEL ---")
model_fused, tokenizer_fused = load(
    "./Qwen3.6-27B-Claude-Opus-Reasoning-Distilled-MLX"
)

print("\n--- GENERATING WITH FUSED MODEL ---")
out_fused = generate(
    model_fused,
    tokenizer_fused,
    prompt=prompt,
    max_tokens=100,
    verbose=True
)
