# Transformers — The Basics

> **Date:** 2026-11-04 · **Updated:** 2026-11-04

## Key Concepts
- **Transformer** — the neural-network architecture behind modern LLMs (GPT, Claude, Llama, Gemini). Introduced in *"Attention Is All You Need"* (Google, 2017).
- **Self-attention** — lets every token look at every other token and decide which ones matter for its meaning.
- **Embedding** — a vector (list of numbers) representing a token's meaning.
- **Positional encoding** — adds word-order information, since attention alone doesn't know order.
- **Context window** — max number of tokens the model can attend to at once.
- **Parameters / weights** — the learned numbers inside the model (billions in an LLM).

## Explanation

### What it is & its role in LLMs
A transformer is the **engine of an LLM**. Its one job: given a sequence of tokens, **predict the next token**. Repeat that token by token and you get generated text.

Before transformers, RNNs/LSTMs read text one word at a time — slow and bad at remembering far-back words. Transformers process **all tokens in parallel** and use **attention** to connect any two words directly, no matter how far apart. That made it possible to train on huge datasets with GPUs → today's LLMs.

### How text flows through it
```text
"The cat sat"
   │ 1. Tokenizer        → token IDs        [464, 3797, 3332]
   │ 2. Embedding        → one vector per token
   │ 3. + Positional info → vectors now know word order
   │ 4. N × Transformer blocks (e.g. 32–100+ layers)
   │      ├─ Self-attention   : tokens exchange information
   │      └─ Feed-forward     : each token's vector is processed further
   │ 5. Output layer     → probability for every token in the vocabulary
   ▼
next token = "on" (sampled) → append and repeat
```

### Self-attention, intuitively
In *"The animal didn't cross the street because **it** was tired"*, attention helps `it` focus on `animal`, not `street`.
Each token makes three vectors:
| Vector | Plain meaning |
|---|---|
| **Query (Q)** | "What am I looking for?" |
| **Key (K)** | "What do I contain?" |
| **Value (V)** | "What info do I pass on if chosen?" |

Score = how well a Query matches each Key → turned into weights → weighted sum of Values.
**Multi-head attention** runs several of these in parallel, each learning different relationships (grammar, meaning, references…).

### Transformer variants
| Type | Sees | Good at | Examples |
|---|---|---|---|
| **Encoder-only** | whole input both ways | understanding, classification, embeddings | BERT |
| **Decoder-only** | only past tokens (causal) | text generation — **most LLMs** | GPT, Claude, Llama |
| **Encoder-decoder** | input fully, output step by step | translation, summarization | T5, original Transformer |

### Why it matters
- **Parallel training** → scales to massive data and model sizes.
- **Long-range understanding** → connects ideas across a whole document.
- **Cost:** attention compares every token with every other → compute grows ~quadratically with context length (why long context is expensive).

## Examples

Scaled dot-product attention (the core formula):

```text
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

Minimal NumPy version:

```python
import numpy as np

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                    # how much each token attends to others
    weights = np.exp(scores) / np.exp(scores).sum(-1, keepdims=True)  # softmax
    return weights @ V                                 # mix values by attention weights
```

## Open Questions
- [ ] How exactly are Q, K, V weight matrices learned during training?
- [ ] What happens inside the feed-forward layers — is that where "knowledge" is stored?

## Related
- [Tokenization — Best Practices](01-tokenization.md) — step 1 of the pipeline
- [Embeddings — The Basics](02-embeddings.md) — step 2 of the pipeline
- [Inference — The Basics](04-inference.md) — how the trained transformer generates answers
