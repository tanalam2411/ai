# Tokenization — Best Practices

> **Date:** 2026-11-04 · **Updated:** 2026-11-04

## Key Concepts
- **Token** — the unit a model actually reads/writes; a word, sub-word, byte or symbol.
- **Vocabulary size** — number of distinct tokens the tokenizer knows (e.g. 32k, 100k, 200k+).
- **Special tokens** — reserved tokens with structural meaning (`<bos>`, `<eos>`, `<pad>`, chat-role markers).
- **Fertility** — average number of tokens produced per word. Lower = more efficient.
- **BPE / Unigram** — common sub-word algorithms that merge frequent character sequences into tokens.

## Explanation

### 1. Vocabulary size matters
| Smaller vocab (~32k) | Larger vocab (~100k–250k) |
|---|---|
| Smaller embedding/output matrices | More parameters in embedding + softmax layers |
| Text splits into more tokens → longer sequences | Fewer tokens per text → cheaper, longer effective context |
| Each token seen often → well-trained embeddings | Rare tokens may be under-trained |
| Weak for multilingual text | Better coverage of many languages, code, symbols |

**Rule of thumb:** pick the vocab size based on the data mix — multilingual / code-heavy models benefit from larger vocabularies.

### 2. Special tokens
- Reserve them up front (`<bos>`, `<eos>`, `<pad>`, `<unk>`, role/tool markers) — adding later means resizing embeddings.
- They must **never** be produced from plain user text (prevents prompt-injection via fake role markers).
- Always use the model's own chat template; wrong or missing special tokens silently degrade output.

### 3. Fertility
- Measures efficiency: `fertility = total tokens / total words`.
- English is often ~1.3 tokens/word; under-represented languages can be 3–5×+ → higher cost, shorter usable context, worse quality.
- **Best practice:** measure fertility per language/domain on your own data before choosing a tokenizer; balance the training corpus for the tokenizer.

### 4. Numbers
- Inconsistent splits (`1234` → `12` + `34`, but `1235` → `123` + `5`) make arithmetic hard for the model.
- **Best practice:** split numbers into single digits (or fixed groups of up to 3 digits) so place value stays consistent.

### 5. Code
- Whitespace and indentation carry meaning (esp. Python) — tokenizers should have tokens for runs of spaces/tabs, not one token per space.
- Common keywords, operators and identifiers (`def`, `return`, `!==`, `self.`) benefit from dedicated tokens.
- Use **byte-level** fallback so any character (emoji, unusual symbols) can be encoded without `<unk>`.

## Examples

Counting tokens and fertility with OpenAI's `tiktoken`:

```python
import tiktoken

enc = tiktoken.get_encoding("o200k_base")
text = "Tokenization affects cost, context length and accuracy."

tokens = enc.encode(text)
print(len(tokens))                                  # token count
print([enc.decode([t]) for t in tokens])            # see the splits
print(len(tokens) / len(text.split()))              # fertility
```

## Open Questions
- [ ] How do different tokenizers' fertility compare for my own languages/domain?
- [ ] Do byte-level / tokenizer-free models remove the need for these trade-offs?

## Related
- [Embeddings — The Basics](02-embeddings.md) — what each token becomes next
- [Transformers — The Basics](03-transformers.md) — what consumes the tokens
