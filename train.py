import os
import mlx_tune.sft_trainer
import mlx_lm.tuner.callbacks
from tensorboardX import SummaryWriter
from datasets import load_from_disk
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig, train_on_responses_only

# 1. Setup TensorBoard logging
class TensorBoardCallback(mlx_lm.tuner.callbacks.TrainingCallback):
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard writer initialized at: {os.path.abspath(log_dir)}")

    def on_train_loss_report(self, train_info: dict):
        iteration = train_info.get("iteration")
        if iteration is not None:
            for metric, value in train_info.items():
                if metric != "iteration" and value is not None:
                    self.writer.add_scalar(f"train/{metric}", value, iteration)
            loss = train_info.get("train_loss", 0.0)
            lr = train_info.get("learning_rate", 0.0)
            print(f"Iter {iteration}: train_loss = {loss:.4f}, lr = {lr:.2e}")

    def on_val_loss_report(self, val_info: dict):
        iteration = val_info.get("iteration")
        if iteration is not None:
            for metric, value in val_info.items():
                if metric != "iteration" and value is not None:
                    self.writer.add_scalar(f"val/{metric}", value, iteration)
            val_loss = val_info.get("val_loss", 0.0)
            print(f"Iter {iteration}: val_loss = {val_loss:.4f}")

# Monkeypatch the SFTTrainer's training function to inject the callback
orig_mlx_train = mlx_tune.sft_trainer.mlx_train

def patched_mlx_train(*args, **kwargs):
    kwargs['training_callback'] = TensorBoardCallback(log_dir="./tb_logs")
    for arg in args:
        if hasattr(arg, 'clear_cache_threshold'):
            arg.clear_cache_threshold = 0
    if 'args' in kwargs:
        kwargs['args'].clear_cache_threshold = 0
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
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

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
        max_steps=50,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adam",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        save_steps=200,
        save_total_limit=1,
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