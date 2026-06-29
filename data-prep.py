from datasets import load_dataset, concatenate_datasets, Dataset
from mlx_lm import load
from mlx_tune import get_chat_template
import re
import json
import multiprocessing as mp
import pandas as pd
 
RANDOM_SEED = 12181531
MAX_CONTEXT_WINDOW = 8192

num_samples_dict = {
    "ds1": 3900,  # nohurry/Opus-4.6-Reasoning-3000x-filtered
    "ds2": 700,  # Jackrong/Qwen3.5-reasoning-700x
    "ds3": 9633,  # Roman1111111/claude-opus-4.6-10000x
}

# Load tokenizer
_, tokenizer = load("mlx-community/Qwen3.6-27B-OptiQ-4bit")
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3",
)

def load_ds3_via_pandas_parquet():
    parquet_path = (
        "hf://datasets/Roman1111111/claude-opus-4.6-10000x"
        "@refs/convert/parquet/default/train/0000.parquet"
    )
    df = pd.read_parquet(parquet_path)
    return Dataset.from_pandas(df, preserve_index=False)

def load_and_sample(dataset_name, sample_count=None, split="train", subset=None):
    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
    except ValueError as e:
        err = str(e)
        if dataset_name == "Roman1111111/claude-opus-4.6-10000x" and "Feature type 'Json' not found" in err:
            ds = load_ds3_via_pandas_parquet()
        else:
            raise

    if sample_count is not None:
        sample_count = min(sample_count, len(ds))
        ds = ds.shuffle(seed=RANDOM_SEED).select(range(sample_count))

    return ds

# ds1: problem / thinking / solution
# ds2: multi-turn conversation
# ds3: messages with possible reasoning fields
ds1 = load_and_sample("nohurry/Opus-4.6-Reasoning-3000x-filtered", num_samples_dict["ds1"], split="train")
ds2 = load_and_sample("Jackrong/Qwen3.5-reasoning-700x", num_samples_dict["ds2"], split="train")
ds3 = load_and_sample("Roman1111111/claude-opus-4.6-10000x", num_samples_dict["ds3"], split="train")


def _strip(x):
    return (x or "").strip()


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def normalize_assistant_to_think_solution(text: str) -> str:
    text = _strip(text)
    if not text:
        return "<think></think>\n"

    m = THINK_BLOCK_RE.search(text)
    if m:
        think_block = m.group(0).strip()
        rest = text[m.end() :].lstrip()
        return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"
    return f"<think></think>\n{text}".rstrip()


def build_assistant_with_reasoning(content: str, reasoning: str = "") -> str:
    content = _strip(content)
    reasoning = _strip(reasoning)
    if "<think>" in content and "</think>" in content:
        return normalize_assistant_to_think_solution(content)
    if reasoning:
        return (
            f"<think>{reasoning}</think>\n{content}"
            if content
            else f"<think>{reasoning}</think>\n"
        )
    return normalize_assistant_to_think_solution(content)


def parse_message_item(m):
    if isinstance(m, dict):
        return m
    if isinstance(m, str):
        s = m.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def format_ds1(examples):
    out = []
    for p, t, s in zip(
        examples.get("problem", []),
        examples.get("thinking", []),
        examples.get("solution", []),
    ):
        p, t, s = _strip(p), _strip(t), _strip(s)
        if not p or not s:
            continue
        assistant = f"<think>{t}</think>\n{s}" if t else f"<think></think>\n{s}"
        out.append(
            [
                {"role": "user", "content": p},
                {"role": "assistant", "content": assistant},
            ]
        )
    return {"messages": out}


def format_ds2(examples):
    out = []
    for conv in examples.get("conversation", []):
        if not conv:
            continue
        cleaned = []
        for m in conv:
            frm = (m.get("from") or "").strip()
            val = m.get("value", "")
            if frm == "human":
                cleaned.append({"role": "user", "content": _strip(val)})
            elif frm == "gpt":
                cleaned.append(
                    {
                        "role": "assistant",
                        "content": normalize_assistant_to_think_solution(val),
                    }
                )
        if len(cleaned) >= 2 and cleaned[-1]["role"] == "assistant":
            out.append(cleaned)
    return {"messages": out}


def format_ds3(examples):
    out = []
    for msgs in examples.get("messages", []):
        if not msgs:
            continue
        parsed = [pm for pm in (parse_message_item(m) for m in msgs) if pm is not None]
        if not parsed:
            continue
        convo = [m for m in parsed if m.get("role") != "system"]
        if len(convo) < 2 or convo[-1].get("role") != "assistant":
            continue
        cleaned = []
        for m in convo:
            role = m.get("role")
            content = m.get("content", "")
            reasoning = m.get("reasoning", "")
            if role == "assistant":
                content = build_assistant_with_reasoning(content, reasoning)
            else:
                content = _strip(content)
            if role in ("user", "assistant") and content is not None:
                cleaned.append({"role": role, "content": content})
        if len(cleaned) >= 2 and cleaned[-1]["role"] == "assistant":
            out.append(cleaned)
    return {"messages": out}


ds1 = ds1.map(format_ds1, batched=True, remove_columns=ds1.column_names)
ds2 = ds2.map(format_ds2, batched=True, remove_columns=ds2.column_names)
ds3 = ds3.map(format_ds3, batched=True, remove_columns=ds3.column_names)

ds1 = ds1.filter(lambda x: x["messages"] is not None and len(x["messages"]) > 0)
ds2 = ds2.filter(lambda x: x["messages"] is not None and len(x["messages"]) > 0)
ds3 = ds3.filter(lambda x: x["messages"] is not None and len(x["messages"]) > 0)

combined_dataset = concatenate_datasets([ds1, ds2, ds3]).shuffle(seed=RANDOM_SEED)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = combined_dataset.map(
    formatting_prompts_func,
    batched=True,
)
num_proc = mp.cpu_count()
_text_tok = getattr(tokenizer, "_tokenizer", getattr(tokenizer, "tokenizer", tokenizer))

def filter_long_sequences_batched(examples):
    texts = examples["text"]
    tokenized = _text_tok(
        texts,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]
    return [len(toks) <= MAX_CONTEXT_WINDOW for toks in tokenized]

dataset = dataset.filter(filter_long_sequences_batched, batched=True, num_proc=num_proc)

def check_assistant_format(examples):
    convos = examples["messages"]
    ok = []
    for convo in convos:
        good = True
        for m in convo:
            if m["role"] == "assistant":
                c = m.get("content", "")
                if "<think>" not in c or "</think>" not in c:
                    good = False
                    break
                if not re.search(r"</think>\n", c):
                    good = False
                    break
        ok.append(good)
    return ok

dataset = dataset.filter(check_assistant_format, batched=True, num_proc=num_proc)

print(f"Final dataset size: {len(dataset)}")
print("Sample from the final dataset:")
print(dataset[0]["text"][:8000])

# --- Save the processed dataset to disk ---
print("\nSaving processed dataset to disk...")
output_dir = "mlx-finetune-data"
dataset.save_to_disk(output_dir)
print(f"✅ Dataset saved to '{output_dir}'")