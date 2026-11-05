# Inference — The Basics

> **Date:** 2026-11-05 · **Updated:** 2026-11-05

## Key Concepts
- **Inference** — using an already-trained model to produce an output for new input. In LLMs: generating a response to a prompt.
- **Training vs inference** — training *learns* the weights (once, very expensive); inference *uses* the frozen weights (every request, millions of times).
- **Prefill** — the model reads the whole prompt in one parallel pass.
- **Decode** — the model generates the answer one token at a time.
- **KV cache** — saved attention data from earlier tokens so they aren't recomputed for each new token.
- **Sampling** — how the next token is picked from the probabilities (temperature, top-p, top-k).
- **Latency / throughput** — how fast one user gets an answer vs how many tokens per second the server handles in total.

## Explanation

### What "inference" means
In everyday English, *to infer* = to reach a conclusion from evidence. In AI, inference is the moment a trained model **applies what it learned** to new data:

| | Training | Inference |
|---|---|---|
| **Goal** | Learn patterns, adjust weights | Use the weights to answer |
| **Weights** | Change | Frozen (read-only) |
| **When** | Once (plus fine-tunes) | Every time someone uses the model |
| **Cost** | Huge, one-off (weeks on thousands of GPUs) | Small per request, but adds up at scale |
| **Analogy** | Studying for an exam | Answering the exam questions |

Every time you chat with ChatGPT or Claude, call an API, or run a model locally, you're running **inference**.

### How an LLM runs inference
```text
Prompt: "Explain embeddings"
  │ 1. Tokenize            → token IDs
  │ 2. Prefill             → whole prompt processed in parallel, KV cache filled
  │ 3. Decode loop:
  │      a. forward pass   → probabilities for every vocab token
  │      b. sample         → pick next token (e.g. "Embeddings")
  │      c. append token, update KV cache
  │      d. repeat until <eos> token or max_tokens reached
  │ 4. Detokenize          → text streamed back to you
  ▼
"Embeddings are vectors that…"
```
This one-token-at-a-time loop is called **autoregressive generation** — it's why answers *stream* word by word.

### Sampling parameters (you control these in APIs)
| Parameter | What it does | Typical use |
|---|---|---|
| **temperature** | Low (0–0.3) = focused, predictable; high (0.8–1+) = creative, random | 0 for facts/code, higher for brainstorming |
| **top_p** | Only sample from the smallest set of tokens whose probabilities add up to *p* | 0.9–1.0 |
| **top_k** | Only sample from the *k* most likely tokens | 40–100 |
| **max_tokens** | Hard limit on output length | Control cost & length |
| **stop sequences** | Stop generating when this text appears | Structured output |

*Greedy decoding* = always pick the most likely token (like temperature 0).

### Performance metrics
- **TTFT (time to first token)** — wait before the answer starts; driven by prefill / prompt length.
- **Tokens per second (TPS)** — streaming speed; driven by decode.
- **Throughput** — total tokens/sec across all users on a server.
- **Context length** — longer prompts = more memory (KV cache) and slower prefill.

### Cost
- APIs charge **per token**, usually per million: **input tokens** (prompt) and **output tokens** (answer, more expensive since decoding is sequential).
- Ways to save: shorter prompts, prompt caching, smaller models for easy tasks, limiting `max_tokens`.

### Where inference runs
| Option | Examples | Trade-off |
|---|---|---|
| **Cloud API** | Anthropic, OpenAI, Google APIs | Easiest, pay per token, no hardware |
| **Self-hosted server** | vLLM, TGI, SGLang on your own GPUs | Control & privacy, you manage infra |
| **Local / on-device** | Ollama, llama.cpp, LM Studio | Free & private, limited by your RAM/GPU |

**Hardware:** GPUs (or TPUs/AI chips) because inference is massive matrix math. The main limit is usually **memory** — the model weights + KV cache must fit (e.g. a 7B model ≈ 14 GB in 16-bit, ≈ 4 GB at 4-bit).

### Common optimizations
- **Quantization** — store weights in fewer bits (16 → 8 → 4) → less memory, faster, small quality loss.
- **KV caching** — reuse past attention results instead of recomputing.
- **Batching** — serve many users' requests together for higher throughput.
- **Speculative decoding** — a small model drafts tokens, the big model verifies several at once.
- **Distillation** — train a smaller "student" model to mimic a big one for cheaper inference.

## Examples

Inference through an API — the parameters above in action:

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=300,          # cap output length & cost
    temperature=0.2,         # focused, factual answer
    messages=[{"role": "user", "content": "Explain embeddings in 2 sentences."}],
)
print(response.content[0].text)
print(response.usage)        # input_tokens / output_tokens → what you pay for
```

Local inference with Ollama:

```bash
ollama run llama3.2 "Explain embeddings in 2 sentences."
```

## Open Questions
- [ ] How does the KV cache actually work in memory, and why does it limit context length?
- [ ] When is self-hosting cheaper than paying for an API?

## Related
- [Tokenization — Best Practices](01-tokenization.md) — step 1 of every inference call
- [Embeddings — The Basics](02-embeddings.md) — tokens become vectors
- [Transformers — The Basics](03-transformers.md) — the model that runs each forward pass
