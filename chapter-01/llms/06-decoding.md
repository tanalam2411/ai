# Decoding — How LLMs Turn Probabilities into Text

> **Date:** 2026-11-12 · **Updated:** 2026-11-12

## Key Concepts
- **Decoding** — the strategy used to pick output tokens from the model's predicted probabilities. Same model + different decoding = very different text.
- **Logits** — raw scores the model outputs for every token in the vocabulary; **softmax** turns them into probabilities.
- **Deterministic decoding** — always picks by a fixed rule (greedy, beam search) → same input, same output.
- **Stochastic decoding (sampling)** — draws randomly from the (reshaped) distribution (temperature, top-k, top-p, min-p).
- **Logit processors** — rules that edit the logits before picking: repetition penalties, logit bias, grammar constraints.
- **Stopping criteria** — when decoding ends: `<eos>` token, stop sequence, or `max_tokens`.

## Explanation

### Where decoding sits
The model itself never "writes text" — at each step it only outputs a probability for every token. **Decoding is the algorithm wrapped around the model** that chooses one token, appends it, and loops (the *decode* phase from [Inference](05-inference.md)).

```text
prompt ──► model ──► logits (one score per vocab token, ~100k numbers)
                        │
                        ▼  logit processors   (repetition penalty, logit bias, grammar mask)
                        ▼  temperature        (sharpen / flatten)
                        ▼  truncation         (top-k, top-p, min-p)
                        ▼  pick a token       (argmax, beam, or random sample)
                        │
                 append token ──► stop? (<eos> / stop sequence / max_tokens) ──► else loop
```

### The two families

| | Deterministic (search) | Stochastic (sampling) |
|---|---|---|
| **Idea** | Find the *most likely* text | Draw a *plausible* text at random |
| **Methods** | Greedy, beam search, contrastive search | Temperature, top-k, top-p, min-p, typical |
| **Output** | Same every run | Varies each run |
| **Good for** | Translation, summarization, short factual answers | Chat, creative writing, brainstorming |
| **Failure mode** | Bland, repetitive loops ("I think I think I think…") | Incoherent or off-topic if too random |

### Decoding methods

**1. Greedy decoding** — always take the single most likely token (`argmax`).
- Fast, simple, deterministic. Same as `temperature = 0`.
- Short-sighted: a locally best token can lead to a worse sentence overall; prone to repetition in long text.

**2. Beam search** — keep the **B** best partial sequences ("beams") at every step, expand each, keep the top B by total log-probability.
- Explores more than greedy → finds higher-probability sentences.
- **Length penalty** stops it from favoring short outputs (each extra token lowers total probability).
- Still bland/repetitive for open-ended text; B× more compute. Used for translation, ASR, classic seq2seq — rarely for chat LLMs.
- Variant: **diverse beam search** — groups of beams penalized for being similar.

```text
Beam width B = 2, prompt "The cat"
step 1: "sat" (0.5)  "is" (0.3)  "ran" (0.2)         → keep: sat, is
step 2: "sat on" 0.5×0.7=0.35   "sat down" 0.5×0.2=0.10
        "is sleeping" 0.3×0.6=0.18  "is black" 0.3×0.3=0.09 → keep: "sat on", "is sleeping"
```

**3. Temperature sampling** — divide logits by T, then sample. Low T = sharp/safe, high T = flat/creative. *(Worked example in [Parameters & Sampling](04-parameters.md).)*

**4. Top-k sampling** — sample only from the k most likely tokens. Fixed size, doesn't adapt to confidence.

**5. Top-p (nucleus) sampling** — sample from the smallest set whose probabilities sum to p. Adapts: small set when confident, larger when unsure. The most common default.

**6. Min-p sampling** — keep tokens with probability ≥ `min_p × (top token's probability)`.
- `min_p = 0.1`, top token 0.60 → cutoff 0.06. If top token is only 0.20 → cutoff 0.02, more options kept.
- Scales with confidence; works well at high temperature. Popular in local tools (llama.cpp, Ollama, vLLM).

**7. Typical sampling** — keep tokens whose "surprise" is close to the average surprise (entropy) of the distribution; avoids both the overly obvious and the very unlikely. Less common.

**8. Contrastive search** — picks a token that is likely **and** dissimilar to what was already said (penalizes hidden-state similarity). Deterministic but less repetitive than greedy.

### Controlling repetition

| Knob | What it does |
|---|---|
| **repetition_penalty** (e.g. 1.1) | Divides logits of tokens that already appeared → less likely to repeat |
| **frequency_penalty** | Penalty grows with how *often* a token appeared |
| **presence_penalty** | Flat penalty if a token appeared *at all* → nudges toward new topics |
| **no_repeat_ngram_size** | Hard-bans repeating any n-gram (e.g. 3 words) |

### Steering the output

- **Logit bias** — add/subtract a fixed value to specific token IDs (ban a word, force "Yes"/"No").
- **Stop sequences** — end generation when a string appears (e.g. `"\n\nUser:"`, `"</answer>"`).
- **Constrained / structured decoding** — a grammar or JSON schema masks out every token that would make the output invalid, so the model *can only* produce valid JSON / a regex match / an enum value. Powers "structured outputs" and reliable tool calls (libraries: Outlines, XGrammar, llama.cpp GBNF).
- **Logprobs** — many APIs return the probability of each chosen token → useful for confidence scores, classification, detecting hallucination-prone spots.

### Decoding at the "whole answer" level
These run decoding several times and pick or combine results:
- **Best-of-N** — sample N answers, keep the one with the highest score (log-prob, reward model, or tests passing).
- **Self-consistency** — sample several chain-of-thought answers, take the **majority vote** on the final answer. Boosts math/reasoning accuracy.
- **Reasoning / "thinking" models** spend extra decoded tokens thinking before answering — more output tokens, better answers (test-time compute).

### Making decoding faster
Decoding is sequential (one token per forward pass), so it's the slow, expensive part of inference.
- **Speculative decoding** — a small *draft* model guesses the next few tokens; the big model checks them all in **one** pass and keeps the ones it agrees with. Same output quality, often 2–3× faster.
- **Medusa / EAGLE / multi-token prediction** — extra heads on the model itself predict several future tokens, verified the same way.
- **KV cache** — avoids recomputing past tokens each step ([Inference](05-inference.md)).

### Which method when?

| Task | Recommended decoding |
|---|---|
| Tool calls, JSON, extraction | Greedy / temperature 0 + **constrained decoding** |
| Code | Temperature 0–0.2; Best-of-N + run tests for hard problems |
| Math / reasoning | Low temperature, or **self-consistency** with several samples |
| Translation / summarization (seq2seq) | Beam search (B = 4–5) or low temperature |
| Chat / Q&A | Temperature ~0.7, top-p 0.9 |
| Creative writing | Temperature 0.9–1.2 + top-p 0.95 or min-p 0.05–0.1, mild repetition penalty |

**For agent builders:** use low temperature + structured outputs for anything a program parses; use sampling + self-consistency/best-of-N when you can afford extra calls and verify the result.

## Examples

Comparing decoding methods with Hugging Face `generate`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
inputs = tok("Once upon a time", return_tensors="pt")

def gen(**kw):
    out = model.generate(**inputs, max_new_tokens=40, **kw)
    return tok.decode(out[0], skip_special_tokens=True)

print(gen(do_sample=False))                                  # greedy
print(gen(do_sample=False, num_beams=4, length_penalty=1.0)) # beam search
print(gen(do_sample=True, temperature=0.8, top_p=0.9))       # nucleus sampling
print(gen(do_sample=True, temperature=1.0, min_p=0.1))       # min-p sampling
print(gen(penalty_alpha=0.6, top_k=4))                       # contrastive search
print(gen(do_sample=True, repetition_penalty=1.2))           # less repetition
```

Self-consistency in a few lines:

```python
from collections import Counter

answers = [ask_llm(question, temperature=0.8) for _ in range(5)]  # 5 sampled answers
final = Counter(extract_final_answer(a) for a in answers).most_common(1)[0][0]
```

## Open Questions
- [ ] How exactly does constrained decoding mask tokens when one JSON key spans several tokens?
- [ ] How much faster is speculative decoding in practice on Ollama / vLLM?
- [ ] Is self-consistency worth the extra cost vs. a single call to a reasoning model?

## Related
- [Parameters, Hyperparameters & Sampling](04-parameters.md) — temperature / top-k / top-p worked example
- [Inference — The Basics](05-inference.md) — the prefill → decode loop decoding runs in
- [Tokenization — Best Practices](01-tokenization.md) — decoding picks *tokens*, not words
- [Knowledge Distillation](08-distillation.md) — another route to cheaper inference
