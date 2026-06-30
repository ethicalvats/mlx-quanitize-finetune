import os
import mlx_tune.sft_trainer
import mlx_lm.tuner.callbacks
from tensorboardX import SummaryWriter
from datasets import load_from_disk
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig, train_on_responses_only
import mlx_lm.tuner.trainer

# Monkeypatch default_loss to prevent NaNs/division-by-zero on fully masked/truncated sequences (autodiff-safe)
def patched_default_loss(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn
    
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ntoks = mask.sum()
    has_targets = ntoks > 0

    # Create an autodiff-safe mask that has at least one active token if ntoks == 0
    safe_mask = mx.where(has_targets, mask, mx.arange(targets.shape[1]) == 0)

    ce = nn.losses.cross_entropy(logits, targets) * safe_mask
    ce_sum = ce.astype(mx.float32).sum()

    # If has_targets is False, return 0.0. The true branch is ce_sum / 1 which is valid and won't produce NaNs in gradients.
    loss = mx.where(has_targets, ce_sum / mx.maximum(ntoks, 1), mx.array(0.0))

    return loss, ntoks

mlx_lm.tuner.trainer.default_loss = patched_default_loss

orig_train = mlx_lm.tuner.trainer.train
def patched_train(*args, **kwargs):
    if 'loss' not in kwargs:
        kwargs['loss'] = patched_default_loss
    return orig_train(*args, **kwargs)
mlx_lm.tuner.trainer.train = patched_train

orig_evaluate = mlx_lm.tuner.trainer.evaluate
def patched_evaluate(*args, **kwargs):
    if 'loss' not in kwargs:
        kwargs['loss'] = patched_default_loss
    return orig_evaluate(*args, **kwargs)
mlx_lm.tuner.trainer.evaluate = patched_evaluate

MAX_STEPS = 350
RESUME_STEPS = 200

# Monkeypatch SFTTrainer._save_adapter_config to write correct num_layers (e.g. 16) instead of model total layers (64)
def patched_save_adapter_config(self):
    import json
    lora_config = {}
    if hasattr(self.model, 'lora_config') and self.model.lora_config:
        lora_config = self.model.lora_config.copy()
    
    r = lora_config.get('r', self.lora_r)
    alpha = lora_config.get('lora_alpha', self.lora_alpha)
    
    # Use the trainer's actual num_layers (e.g. 16) instead of model's total layers (64)
    num_layers = self.num_layers if self.num_layers is not None else 64
    
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": r,
            "scale": alpha / r,
            "dropout": lora_config.get('lora_dropout', 0.0),
        },
    }
    
    if hasattr(self.model, '_resolved_lora_keys') and self.model._resolved_lora_keys:
        adapter_config["lora_parameters"]["keys"] = self.model._resolved_lora_keys
    else:
        target_modules = lora_config.get('target_modules', [])
        if target_modules:
            from mlx_tune.model import _resolve_target_modules
            model_for_resolve = self.model.model if hasattr(self.model, 'model') else self.model
            keys = _resolve_target_modules(model_for_resolve, target_modules)
            adapter_config["lora_parameters"]["keys"] = keys
            
    config_path = self.adapter_path / "adapter_config.json"
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)
    print(f"  [Patched] Adapter config saved to: {config_path} with num_layers={num_layers}")

SFTTrainer._save_adapter_config = patched_save_adapter_config

# 1. Setup TensorBoard logging
class TensorBoardCallback(mlx_lm.tuner.callbacks.TrainingCallback):
    def __init__(self, log_dir="runs", resume_steps=0):
        self.writer = SummaryWriter(log_dir)
        self.resume_steps = resume_steps
        print(f"TensorBoard writer initialized at: {os.path.abspath(log_dir)} (resuming from step {resume_steps})")

    def on_train_loss_report(self, train_info: dict):
        iteration = train_info.get("iteration")
        if iteration is not None:
            plot_iter = iteration + self.resume_steps
            for metric, value in train_info.items():
                if metric != "iteration" and value is not None:
                    self.writer.add_scalar(f"train/{metric}", value, plot_iter)
            loss = train_info.get("train_loss", 0.0)
            lr = train_info.get("learning_rate", 0.0)
            print(f"Iter {plot_iter}: train_loss = {loss:.4f}, lr = {lr:.2e}")

    def on_val_loss_report(self, val_info: dict):
        iteration = val_info.get("iteration")
        if iteration is not None:
            plot_iter = iteration + self.resume_steps
            for metric, value in val_info.items():
                if metric != "iteration" and value is not None:
                    self.writer.add_scalar(f"val/{metric}", value, plot_iter)
            val_loss = val_info.get("val_loss", 0.0)
            print(f"Iter {plot_iter}: val_loss = {val_loss:.4f}")

# Monkeypatch the SFTTrainer's training function to inject the callback
orig_mlx_train = mlx_tune.sft_trainer.mlx_train

def patched_mlx_train(*args, **kwargs):
    kwargs['training_callback'] = TensorBoardCallback(log_dir="./tb_logs", resume_steps=RESUME_STEPS)
    
    # Extract and modify training args
    training_args = kwargs.get('args') or (args[4] if len(args) > 4 else None)
    if training_args is not None:
        training_args.clear_cache_threshold = 0
        training_args.iters = MAX_STEPS - RESUME_STEPS
        print(f"Adjusting training iterations to run {training_args.iters} steps (stopping at step {MAX_STEPS})")
        
    for arg in args:
        if hasattr(arg, 'clear_cache_threshold'):
            arg.clear_cache_threshold = 0
            
    # Offset learning rate scheduler for resuming and inject gradient clipping
    optimizer = kwargs.get('optimizer') or (args[1] if len(args) > 1 else None)
    if optimizer is not None:
        if RESUME_STEPS > 0:
            import mlx.core as mx
            original_lr_schedule = optimizer._schedulers.get('learning_rate')
            if original_lr_schedule is not None and callable(original_lr_schedule):
                print(f"Offsetting learning rate schedule by {RESUME_STEPS} steps")
                optimizer._schedulers['learning_rate'] = lambda step: original_lr_schedule(step + RESUME_STEPS)
                # Evaluate initial LR array value for the starting step
                optimizer.learning_rate = mx.array(original_lr_schedule(RESUME_STEPS), dtype=mx.float32)
                
        # Inject gradient clipping to stabilize training
        orig_update = optimizer.update
        def patched_update(model, gradients):
            from mlx.utils import tree_flatten, tree_map
            import mlx.core as mx
            
            grads = [v for _, v in tree_flatten(gradients)]
            if grads:
                sum_squares = sum(mx.sum(g ** 2) for g in grads)
                global_norm = mx.sqrt(sum_squares)
                max_norm = 1.0
                scale = max_norm / mx.maximum(global_norm, 1e-6)
                scale = mx.minimum(scale, 1.0)
                gradients = tree_map(lambda x: x * scale, gradients)
                
            return orig_update(model, gradients)
            
        optimizer.update = patched_update
        print("Successfully injected gradient clipping (max_norm=1.0) into the optimizer.")
            
    return orig_mlx_train(*args, **kwargs)

mlx_tune.sft_trainer.mlx_train = patched_mlx_train

# 2. Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Qwen3.6-27B-4bit",
    max_seq_length=512,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Resume training from existing adapters if they exist
# Prioritize the 16-layer step 200 directory, fall back to the original adapters if needed
adapter_dir = "./lora_finetuned/adapters_16layers_step200"
if not os.path.exists(os.path.join(adapter_dir, "adapters.safetensors")):
    adapter_dir = "./lora_finetuned/adapters"

if os.path.exists(os.path.join(adapter_dir, "adapters.safetensors")):
    print(f"Resuming training: Loading existing adapters from {adapter_dir}")
    # Freeze the base model first to ensure only LoRA parameters are trainable
    model.model.freeze()
    model.load_adapter(adapter_dir)

# 3. Load dataset
dataset = load_from_disk("mlx-finetune-data")
if "text" in dataset.column_names:
    dataset = dataset.remove_columns(["text"])
eval_dataset = dataset.select([0])

# 4. Initialize Trainer
drive_output_path = "./lora_finetuned"

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=36,
        max_seq_length=512,
        num_layers=16,
        grad_checkpoint=True,
        warmup_ratio=0.03,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        learning_rate=3e-5,
        logging_steps=5,
        optim="adam",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        save_steps=50,
        save_total_limit=5,
        save_strategy="steps",
        output_dir=drive_output_path,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n<think>",
)

# 5. Start Training
print("Starting training...")
trainer_stats = trainer.train()