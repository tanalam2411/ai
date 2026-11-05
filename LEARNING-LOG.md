# 📓 Learning Log

Dated session entries, newest first.

<!-- Entry format:
## YYYY-MM-DD
**Learned:** …
**Confused by:** …
**Next:** …
-->

## 2026-11-05
**Learned:**
- [Inference — the basics](llms/04-inference.md) — training vs inference, prefill & decode, KV cache, sampling (temperature, top-p), TTFT & tokens/sec, per-token cost, API vs self-hosted vs local, quantization.
- [In-context learning](prompting/01-in-context-learning.md) — learning from examples in the prompt without changing weights, zero/one/few-shot, ICL vs fine-tuning, few-shot best practices.

**Confused by:** —

**Next:** Try local inference with Ollama; compare zero-shot vs few-shot on the same task; chain-of-thought prompting.

## 2026-11-04
**Learned:**
- Set up this learning journal — notes by topic, a session log, and an auto-maintained index.
- [Tokenization best practices](llms/01-tokenization.md) — vocabulary size trade-offs, special tokens, fertility, number splitting, code tokenization.
- [Transformers — the basics](llms/03-transformers.md) — what a transformer is, its role in LLMs, self-attention (Q/K/V), encoder vs decoder variants.
- [Embeddings — the basics](llms/02-embeddings.md) — vectors of meaning, cosine similarity, static vs contextual, token vs text embeddings, role in RAG.

**Confused by:** —

**Next:** Compare tokenizer fertility on real text; go deeper on attention; explore RAG and vector databases.
