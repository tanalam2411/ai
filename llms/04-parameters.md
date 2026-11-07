# Parameters, Hyperparameters & Sampling Settings

> **Date:** 2026-11-07 · **Updated:** 2026-11-07

## Key Concepts
- **Parameters** — the numbers *inside* the model that it learns during training (weights + biases). "Llama 3 8B" = ~8 billion parameters.
- **Weight** — how strongly one input value influences a neuron's output (a multiplier).
- **Bias** — a constant added to a neuron's output so it can shift up/down regardless of input.
- **Hyperparameters** — settings *you* choose before/for training (learning rate, batch size, epochs, number of layers). Not learned.
- **Sampling (generation) settings** — knobs used at inference time to pick the next token: **temperature, top-k, top-p**. Often loosely called "parameters" in APIs, but they don't change the model.

## Explanation

### Parameters = what the model "knows"
A neural network layer computes, for each neuron:

```text
output = activation( w1·x1 + w2·x2 + … + wn·xn + b )
                     └──────── weights ────────┘  └ bias
```

- Training starts with random weights/biases and nudges them (gradient descent) billions of times until predictions get good.
- After training, all the model's "knowledge" — grammar, facts, coding patterns — is stored **only** as these numbers. There is no database inside.
- In a transformer, most parameters live in the **attention matrices** (Q/K/V/output projections), the **feed-forward layers**, and the **embedding table**.

**Size → memory:** each parameter is stored as a number (16-bit = 2 bytes).

| Model | Params | Memory (16-bit) | Memory (4-bit) |
|---|---|---|---|
| Small | 1B | ~2 GB | ~0.5 GB |
| Mid | 7–8B | ~14–16 GB | ~4–5 GB |
| Large | 70B | ~140 GB | ~35–40 GB |

More parameters ≈ more capacity to learn, but also more memory, slower and pricier inference.

### The three kinds of "knobs" — don't mix them up

| | Parameters | Hyperparameters | Sampling settings |
|---|---|---|---|
| **Examples** | weights, biases | learning rate, batch size, epochs, LoRA rank | temperature, top-k, top-p, max_tokens |
| **Who sets them** | Learned by training | ML engineer, before training | You, on each API call |
| **When** | Training | Training / fine-tuning | Inference |
| **Change the model?** | They *are* the model | Shape how it's trained | ❌ No — only how output is picked |

### Do parameters matter to AI users and agent builders?
Yes — even though you never edit weights when calling an API:

| You are… | How parameters show up |
|---|---|
| **Choosing a model** | Param count is a rough proxy for capability, cost and speed (e.g. small model for classification, big model for reasoning). |
| **Running locally / self-hosting** | Param count × bytes per param = will it fit on your GPU/RAM? Drives quantization choices. |
| **Calling an API / building an agent** | You tune **sampling settings** per task: temperature 0 for tool calls, JSON and code; higher for brainstorming. |
| **Customizing a model** | Fine-tuning / LoRA *does* change parameters → see [Fine-Tuning & PEFT](06-fine-tuning-peft.md). |

So: weights/biases matter mostly for training and model choice; sampling settings matter every day for users and agents.

### Sampling settings in depth
At each step the model outputs a **probability for every token** in its vocabulary. Sampling settings decide which token is picked.

Example — next token after `"The sky is"`:

| Token | Raw prob |
|---|---|
| blue | 0.60 |
| clear | 0.20 |
| dark | 0.10 |
| falling | 0.07 |
| purple | 0.03 |

**Temperature** — reshapes the distribution (divides the logits by T before softmax).
- `T → 0` → almost always "blue" (greedy, deterministic-ish).
- `T = 1` → probabilities as-is.
- `T > 1` → flatter; "purple" gets a real chance → more creative / more errors.

**Top-k** — keep only the *k* most likely tokens, renormalize, sample.
- `k = 2` → only {blue, clear} can be picked.
- Fixed size — doesn't adapt when the model is very sure or very unsure.

**Top-p (nucleus)** — keep the smallest set whose probabilities add up to *p*.
- `p = 0.9` → blue + clear + dark = 0.90 → {blue, clear, dark}.
- Adapts: if the model is confident (blue = 0.95), the set shrinks to 1 token.

Typical order inside the sampler: **temperature → top-k → top-p → sample**.

| Task | Suggested settings |
|---|---|
| Code, JSON, tool calls, extraction | temperature 0–0.2 |
| General chat / Q&A | temperature ~0.7, top-p 0.9–1.0 |
| Brainstorming, creative writing | temperature 0.9–1.2, top-p 0.95 |

Tips: usually tune **either** temperature **or** top-p, not both (some APIs only allow one). Temperature 0 is not *perfectly* deterministic on most servers.

## Examples

Same prompt, different sampling settings:

```python
import anthropic

client = anthropic.Anthropic()

def ask(**sampling):
    r = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Give a name for a coffee shop."}],
        **sampling,
    )
    return r.content[0].text

print(ask(temperature=0))      # same safe answer almost every time
print(ask(temperature=1))      # more varied names
print(ask(top_k=5))            # only the 5 likeliest tokens at each step
```

Counting parameters of a model yourself:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
print(sum(p.numel() for p in model.parameters()))   # ~494,000,000
```

## Open Questions
- [ ] How is the parameter count split between attention, feed-forward and embeddings in a real model?
- [ ] Why does temperature 0 still give slightly different answers sometimes?

## Related
- [Transformers — The Basics](03-transformers.md) — where the weights live
- [Inference — The Basics](05-inference.md) — the generation loop where sampling happens
- [Fine-Tuning & PEFT (LoRA)](06-fine-tuning-peft.md) — changing parameters after pre-training
