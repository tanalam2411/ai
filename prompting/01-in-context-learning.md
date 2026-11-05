# In-Context Learning (ICL)

> **Date:** 2026-11-05 · **Updated:** 2026-11-05

## Key Concepts
- **In-context learning** — an LLM learning a task *from the prompt itself* (instructions + examples), with **no change to its weights**.
- **Context** — everything the model sees in one request: system prompt, instructions, examples, documents, chat history.
- **Zero-shot** — task described, no examples.
- **One-shot / few-shot** — task shown with 1 or a few input → output examples ("shots").
- **Demonstrations** — the examples given in the prompt.
- **Fine-tuning** — the alternative: actually retraining the weights on many examples.

## Explanation

### What it is
Normally a model "learns" by **training** (weights change). LLMs can also adapt **at inference time**: put a few examples in the prompt and the model picks up the pattern and continues it — like showing a new colleague two filled-in forms and saying "do the rest like this".

The learning is **temporary**: it only exists inside that one context window. Next request, the model has forgotten it.

It was popularised by the GPT-3 paper *"Language Models are Few-Shot Learners"* (2020) — the bigger the model, the better it got at this.

### Zero-shot vs few-shot
```text
Zero-shot:
  Classify the sentiment: "The battery dies in an hour." →

Few-shot:
  "I love this phone!"            → positive
  "Screen cracked on day one."    → negative
  "Works fine, nothing special."  → neutral
  "The battery dies in an hour."  →
```
Few-shot shows the **format, labels and style** you want, so output becomes more consistent.

### ICL vs fine-tuning
| | In-context learning | Fine-tuning |
|---|---|---|
| **Weights** | Unchanged | Updated |
| **Data needed** | A handful of examples | Hundreds–thousands |
| **Speed to try** | Instant — edit the prompt | Hours/days + training cost |
| **Persistence** | Only for that request | Permanent in the model |
| **Cost per request** | Higher (examples use tokens every call) | Lower (shorter prompts) |
| **Best for** | Prototyping, changing tasks, format/style | Stable, high-volume, specialised tasks |

### Why it works (intuition)
- During pre-training the model saw countless patterns like "Q: … A: …", lists, tables, translations.
- Attention lets each new token look back at the examples in the context, so the model **matches the pattern** and continues it.
- Exact mechanism is still researched (e.g. "induction heads" that copy patterns forward).

### Best practices
- **Quality > quantity** — 2–5 clear, correct examples usually beat many messy ones.
- **Be diverse** — cover different cases (all labels, edge cases) so the model doesn't copy one example too closely.
- **Keep a consistent format** — same separators and structure in every example and in the final query.
- **Watch the order** — models can be biased toward the last examples or the most frequent label; shuffle/balance them.
- **Combine with instructions** — say what the task is *and* show examples.
- **Mind the context window & cost** — every example is sent (and paid for) on every call.

### Related techniques built on ICL
- **Chain-of-thought (CoT)** — examples include step-by-step reasoning, so the model reasons before answering.
- **RAG** — retrieved documents are put into the context so the model can use knowledge it wasn't trained on.
- **System prompts / personas** — instructions in context that shape behaviour for the whole conversation.

## Examples

Few-shot prompt via the API — examples passed as earlier chat turns:

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=10,
    system="Classify review sentiment as positive, negative or neutral. Reply with one word.",
    messages=[
        {"role": "user", "content": "I love this phone!"},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": "Screen cracked on day one."},
        {"role": "assistant", "content": "negative"},
        {"role": "user", "content": "The battery dies in an hour."},
    ],
)
print(response.content[0].text)   # → negative
```

## Open Questions
- [ ] How many examples are "enough" before fine-tuning becomes the better choice?
- [ ] Is the model really *learning* in context, or just recognising a pattern it already knows?

## Related
- [Inference — The Basics](../llms/04-inference.md) — ICL happens at inference time, weights frozen
- [Transformers — The Basics](../llms/03-transformers.md) — attention is what lets the model use the examples
- [Tokenization — Best Practices](../llms/01-tokenization.md) — examples cost tokens on every call
