# 📓 Learning Log

Dated session entries, newest first.

<!-- Entry format:
## YYYY-MM-DD
**Learned:** …
**Confused by:** …
**Next:** …
-->

## 2026-11-07
**Learned:**
- [Parameters, hyperparameters & sampling](llms/04-parameters.md) — weights & biases are the learned numbers that *are* the model; hyperparameters are set by humans for training; temperature / top-k / top-p only change how the next token is picked at inference.
- [Fine-tuning & PEFT](llms/06-fine-tuning-peft.md) — full fine-tuning vs PEFT, adapter layers, LoRA (`W + B·A`), QLoRA, and a full PEFT example: base model + JSONL dataset in → small adapter file out.
- [Knowledge distillation](llms/07-distillation.md) — a small student model trained to imitate a big teacher; cheaper, faster, runs locally. Anyone can do it (teacher-generated data + LoRA); useful for agents (distilled tool-calling, routing, sub-agents).

**Confused by:**
- "How parameters come in the picture of AI users or agents, or is it only useful for model training?"
- "Can [distillation] be used by normal person? … or is it just for specific use case like model child model training, or can it be used on agent development?"

**Next:** Run a small LoRA fine-tune on Colab; try distilling a small model for one narrow task; compare it against few-shot prompting on the same task; explore RAG.

## 2026-11-05
**Learned:**
- [Inference — the basics](llms/05-inference.md) — training vs inference, prefill & decode, KV cache, sampling (temperature, top-p), TTFT & tokens/sec, per-token cost, API vs self-hosted vs local, quantization.
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
