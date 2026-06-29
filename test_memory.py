import os
import mlx.core as mx
import mlx.nn as nn
from mlx_tune import FastLanguageModel

# We will test both GC = True and GC = False for the full model!

def test_full_model(use_gc):
    print(f"\n--- Testing Full Model with use_gradient_checkpointing={use_gc} ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Qwen3.6-27B-4bit",
        max_seq_length=1024,
    )
    actual_model = model.model if hasattr(model, 'model') else model

    # Apply PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=use_gc,
    )

    if hasattr(model, '_apply_lora') and not model._lora_applied:
        model._apply_lora()

    # If use_gc is True, mlx_tune's SFTTrainer or trainer wrapper would call grad_checkpoint on layers.
    # Let's manually apply gradient checkpointing to Qwen3_5 layers if use_gc is True, to match the trainer behavior.
    if use_gc:
        from mlx_lm.tuner.trainer import grad_checkpoint
        grad_checkpoint(actual_model.layers[0])

    actual_model.train()
    dummy_input = mx.array([[10, 20, 30, 40] * 256], dtype=mx.int32) # length 1024
    
    def dummy_loss(m, x):
        l = m(x)
        return l.sum()

    loss_and_grad = nn.value_and_grad(actual_model, dummy_loss)
    try:
        (loss_val, grads) = loss_and_grad(actual_model, dummy_input)
        mx.eval(loss_val, grads)
        print(f"✅ Full model backprop with GC={use_gc} SUCCESSFUL!")
    except Exception as e:
        print(f"❌ Full model backprop with GC={use_gc} FAILED: {e}")

# Run only the GC=True test
test_full_model(use_gc=True)
