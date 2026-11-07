# Knowledge Distillation (LLM Distillation)

> **Date:** 2026-11-07 · **Updated:** 2026-11-07

## Key Concepts
- **Distillation** — training a small **student** model to imitate a large **teacher** model, so you get much of the teacher's quality at a fraction of the size, cost and latency.
- **Teacher** — the big, capable model (e.g. a frontier API model or a 70B open model).
- **Student** — the smaller model being trained (e.g. 1–8B). Its weights change; the teacher's don't.
- **Soft labels / logits** — the teacher's full probability distribution over next tokens ("70% *Paris*, 5% *France*…"). Richer signal than just the right answer.
- **Hard labels** — only the teacher's final output text.
- **Synthetic data distillation** — use the teacher to *generate* a dataset (Q&A, reasoning traces, tool calls), then fine-tune the student on it. The most common form today.
- **Reasoning distillation** — training a student on the teacher's step-by-step reasoning so the small model learns to "think" too (e.g. the DeepSeek-R1 distilled models).

## Explanation

### What is it?
Distillation is **a special kind of [fine-tuning](06-fine-tuning-peft.md)** where the training data comes from another model instead of humans.

```
   Prompts ──▶ 🧠 Teacher (big, slow, expensive) ──▶ answers / probabilities
                                                        │
                                                        ▼
                         🐣 Student (small, fast, cheap) trained to copy them
```

| Type | What the student learns from | Needs teacher access to… | Who does it |
|---|---|---|---|
| **Logit (white-box) distillation** | Teacher's token probabilities | Internal logits | Model labs, open-weight teachers |
| **Response (black-box) distillation** | Teacher's generated text | Just an API / chat output | **Anyone** — most practical |
| **Reasoning distillation** | Teacher's chain-of-thought + answer | Text output with reasoning | Labs and hobbyists |

Why soft labels help: if the teacher says "cat 80%, kitten 15%, dog 1%", the student learns *cat and kitten are similar* — something a one-hot "cat" label never teaches.

### Purpose — why do it?

| Goal | How distillation helps |
|---|---|
| 💸 **Lower cost** | Small model = cheaper [inference](05-inference.md) per token |
| ⚡ **Lower latency** | Fewer [parameters](04-parameters.md) → faster responses |
| 📱 **Run locally / on-device** | Fits on a laptop, phone or edge device; private, offline |
| 🎯 **Specialize** | Student becomes an expert at *one* task, often matching the teacher there |
| 🏭 **Scale** | Run millions of requests that would be too pricey on the big model |

Trade-off: the student is great at what it was trained on but **loses the teacher's broad general knowledge**. It's a specialist, not a replacement.

Famous examples: DistilBERT (40% smaller, ~97% of BERT's quality), DeepSeek-R1-Distill-Qwen/Llama, and many "mini" / "small" models labs ship are distilled from their larger siblings.

### Can a normal person use it? — Yes

You don't need a lab. The practical recipe is **synthetic-data distillation + [LoRA/QLoRA](06-fine-tuning-peft.md)**:

1. Collect ~500–5,000 realistic prompts for your task.
2. Send them to a strong model (API) → save the best answers (filter / review bad ones).
3. Fine-tune a small open model (e.g. 1–8B) on those pairs with LoRA on Colab or one GPU.
4. Evaluate student vs teacher on a held-out set.
5. Serve the student cheaply (Ollama, vLLM, llama.cpp).

> ⚠️ **Check terms of service.** Many commercial APIs forbid using their outputs to train *competing* models. Using open-weight teachers (Llama, Qwen, DeepSeek, Mistral…) with permissive licenses avoids this. Teacher mistakes are also copied — filter data.

### Use cases — not just "child model training"

| Use case | Example |
|---|---|
| Classification / extraction | Teacher labels 10k support tickets → student classifies them in milliseconds |
| Structured output | Student reliably emits your exact JSON schema |
| Domain assistant | Teacher writes Q&A on your docs → small in-house model (pair with RAG for facts) |
| Local / private apps | Distilled model running offline on a laptop or phone |
| Cheap reasoning | Small model trained on teacher reasoning traces for math/code |

### Distillation for agent development 🤖

Yes — this is one of the hottest use cases. Agents make **many** LLM calls per task (plan, pick tool, parse result, retry…), so cost and latency multiply.

- **Distill tool-calling** — log the big model's successful agent runs (prompt → tool call → args), train a small model to make the same calls.
- **Distill sub-agents** — keep the big model as the planner/orchestrator, give routine steps (routing, classification, summarizing tool output, extracting fields) to small distilled models.
- **Trajectory distillation** — record full successful multi-step agent traces from the teacher and train the student on them.
- **Router models** — a tiny distilled model decides "easy → small model, hard → big model".

```
User ─▶ Big model (planner) ─┬─▶ Small distilled model: route / classify
                             ├─▶ Small distilled model: call tool X with JSON args
                             └─▶ Small distilled model: summarize tool output
```

Typical path: **prototype with a big model → log good runs → distill the repetitive parts → keep the big model only where it's needed.**

### Distillation vs related ideas

| | Distillation | Fine-tuning | Quantization | Prompting |
|---|---|---|---|---|
| Makes a *smaller* model? | ✅ different, smaller architecture | ❌ same size | ✅ same model, lower-precision numbers | ❌ |
| Training needed? | ✅ | ✅ | ❌ (mostly) | ❌ |
| Data from | Teacher model | Humans / any dataset | — | — |

They combine: distill → fine-tune with LoRA → quantize → run locally.

## Examples

Building a distillation dataset from a teacher (black-box, synthetic data):

```python
import json
from openai import OpenAI  # any OpenAI-compatible endpoint, e.g. a local/open-weight teacher

teacher = OpenAI(base_url="http://localhost:8000/v1", api_key="x")  # e.g. vLLM serving Llama-70B
prompts = [l.strip() for l in open("prompts.txt")]

with open("distill.jsonl", "w") as f:
    for p in prompts:
        r = teacher.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": p}],
            temperature=0.2,
        )
        f.write(json.dumps({"messages": [
            {"role": "user", "content": p},
            {"role": "assistant", "content": r.choices[0].message.content},
        ]}) + "\n")
# Next: LoRA fine-tune a 1–8B student on distill.jsonl (see the PEFT note)
```

Classic logit-distillation loss (what labs use when they have the teacher's logits):

```python
import torch.nn.functional as F

T = 2.0  # temperature softens both distributions
soft = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction="batchmean",
) * T * T
hard = F.cross_entropy(student_logits, labels)
loss = 0.5 * soft + 0.5 * hard
```

## Open Questions
- [ ] Try it: distill a small model for one narrow task and measure quality vs the teacher.
- [ ] How much data is "enough" for a good student on a narrow task?
- [ ] Distillation for agents: how to evaluate a distilled tool-calling model reliably?

## Related
- [Fine-Tuning & PEFT (LoRA)](06-fine-tuning-peft.md) — the training method used to build the student
- [Parameters, Hyperparameters & Sampling](04-parameters.md) — why fewer parameters = cheaper; temperature in the loss
- [Inference — The Basics](05-inference.md) — the cost/latency distillation reduces; quantization
- [In-Context Learning](../prompting/01-in-context-learning.md) — try prompting before training anything
