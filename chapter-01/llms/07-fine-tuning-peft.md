# Fine-Tuning & PEFT (LoRA, Adapters)

> **Date:** 2026-11-07 · **Updated:** 2026-11-07

## Key Concepts
- **Pre-training** — training a model from scratch on huge text data → a general "base model".
- **Fine-tuning** — continuing training on a smaller, task-specific dataset so the model gets better at *your* task/style/format.
- **Full fine-tuning** — update **all** parameters. Best quality ceiling, but needs lots of GPU memory and stores a full model copy per task.
- **PEFT (Parameter-Efficient Fine-Tuning)** — freeze the base model and train only a **small number of extra parameters** (often <1%).
- **Adapter layers** — small trainable modules inserted inside a frozen model.
- **LoRA (Low-Rank Adaptation)** — the most popular PEFT method: adds two small matrices next to existing weight matrices.
- **QLoRA** — LoRA on top of a 4-bit quantized base model → fine-tune 7–8B models on a single consumer GPU.

## Explanation

### Why fine-tune at all?
Options for customizing a model, cheapest first:

| Approach | Changes weights? | Use when |
|---|---|---|
| [Prompting / few-shot](../prompting/01-in-context-learning.md) | ❌ | Try this first |
| RAG | ❌ | Model needs *your facts / documents* |
| **PEFT (LoRA)** | ✅ small add-on | Model needs a consistent *behaviour, style, format or skill* |
| Full fine-tuning | ✅ all | Big shift, lots of data & compute |

Rule of thumb: **fine-tuning teaches *how* to answer; RAG gives *what* to know.**

### Full fine-tuning vs PEFT

| | Full fine-tuning | PEFT / LoRA |
|---|---|---|
| Trainable params (7B model) | 7 billion | ~5–40 million |
| GPU memory | 100+ GB (multi-GPU) | ~6–16 GB (QLoRA) |
| Output size | Full model (~14 GB) | Adapter (~10–100 MB) |
| Multiple tasks | One full copy per task | One base + swap small adapters |
| Risk of forgetting old skills | Higher | Lower (base is frozen) |

### Adapter layers
Original idea (2019): insert tiny bottleneck networks inside each transformer block; train only those.

```text
… → [frozen layer] → [adapter: down-project → activation → up-project] → …
```
Downside: extra layers add a little inference latency. LoRA avoids that.

### LoRA — how it works
A weight matrix `W` (say 4096 × 4096 ≈ 16.7M params) is **frozen**. LoRA learns its *change* as two thin matrices:

```text
W' = W + ΔW,   ΔW = B · A
           A: r × 4096,  B: 4096 × r,   r = rank (e.g. 8)

Trainable: 2 × 4096 × 8 = 65,536  (≈ 0.4% of 16.7M)
```

- Idea: the update needed for a new task is "low-rank" — it can be described by far fewer numbers.
- `B` starts at zero → training begins exactly at the base model.
- After training, `B·A` can be **merged into W** → zero extra inference latency.
- Usually applied to attention projections (`q_proj, k_proj, v_proj, o_proj`), sometimes the feed-forward layers too.

**Key LoRA hyperparameters**

| Hyperparameter | Meaning | Typical |
|---|---|---|
| `r` (rank) | Size of the low-rank matrices; higher = more capacity | 8–64 |
| `lora_alpha` | Scaling of the update (`alpha / r`) | 16–32 (often 2×r) |
| `target_modules` | Which weight matrices get LoRA | attention projections |
| `lora_dropout` | Regularization | 0.05 |
| learning rate | Step size for training | 1e-4 – 2e-4 |
| epochs | Passes over the data | 1–3 |

Other PEFT methods (same family): prefix tuning, prompt tuning, IA³ — LoRA/QLoRA is the default choice today.

## Examples

### PEFT end to end — what you need, input, output
Goal: make a small open model reply to support tickets in **your company's tone and JSON format**.

**What you need**

| Item | Example |
|---|---|
| Base model (open weights) | `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` |
| Dataset | 500–5,000 high-quality input → output examples |
| Hardware | 1 GPU (Colab T4/A100, or a 16–24 GB card); QLoRA for 7–8B |
| Libraries | `transformers`, `peft`, `trl`, `datasets` (+ `bitsandbytes` for QLoRA) |
| Eval set | ~50–100 held-out examples to check before/after |

**Input — training data** (`train.jsonl`, one example per line, chat format):

```json
{"messages": [{"role": "user", "content": "My order #123 hasn't arrived."}, {"role": "assistant", "content": "{\"intent\": \"shipping_delay\", \"reply\": \"Sorry about the wait! Order #123 is with the courier and should arrive within 2 days.\"}"}]}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "{\"intent\": \"account\", \"reply\": \"No problem — click 'Forgot password' on the login page and follow the email link.\"}"}]}
```

**Training script** (LoRA with Hugging Face):

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("json", data_files="train.jsonl", split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",       # frozen base model
    train_dataset=dataset,
    peft_config=peft_config,                  # only LoRA weights are trained
    args=SFTConfig(
        output_dir="support-lora",
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
    ),
)
trainer.model.print_trainable_parameters()   # e.g. trainable: 4.4M / 1.5B (0.29%)
trainer.train()
trainer.save_model("support-lora")
```

**Output — a small adapter folder** (not a whole new model):

```text
support-lora/
├── adapter_config.json          # r, alpha, target modules, base model name
└── adapter_model.safetensors    # the trained A/B matrices (~10–20 MB)
```

**Using it** — load base model + adapter (or merge them into one model):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "support-lora")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

msgs = [{"role": "user", "content": "I was charged twice."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=80)[0][ids.shape[1]:]))
# → {"intent": "billing", "reply": "Sorry about that! ..."}

merged = model.merge_and_unload()            # optional: bake LoRA into the weights
merged.save_pretrained("support-model")
```

**Summary**

```text
INPUT:  base model (frozen) + dataset of examples + LoRA hyperparameters
          │  train only A/B matrices (<1% of params)
          ▼
OUTPUT: adapter file (MBs) → load on top of base, or merge into a new model
```

No GPU? Managed options (OpenAI / Google / AWS Bedrock fine-tuning APIs, Together, Unsloth notebooks on Colab) take the same JSONL and return a fine-tuned model ID.

## Open Questions
- [ ] How do I choose the rank `r` — when is 8 enough vs 64?
- [ ] How many examples are needed before LoRA beats few-shot prompting?
- [ ] How to evaluate whether the fine-tuned model actually got better (and didn't forget other skills)?

## Related
- [Parameters, Hyperparameters & Sampling](04-parameters.md) — what's being trained, and which knobs are hyperparameters
- [Transformers — The Basics](03-transformers.md) — the attention matrices LoRA attaches to
- [Inference — The Basics](05-inference.md) — quantization, used by QLoRA
- [In-Context Learning](../prompting/01-in-context-learning.md) — the no-training alternative
