# Embeddings — The Basics

> **Date:** 2026-11-04 · **Updated:** 2026-11-04

## Key Concepts
- **Embedding** — a list of numbers (a vector) that represents the *meaning* of a token, word, sentence or document.
- **Dimension** — length of the vector (e.g. 768, 1536, 3072). Each dimension captures some learned feature of meaning.
- **Vector space** — the "map" where embeddings live; similar meanings sit close together.
- **Cosine similarity** — the standard way to measure how close two embeddings are (1 = same direction, 0 = unrelated).
- **Token embedding** — vector the LLM looks up for each token (step 2 of the transformer).
- **Text / sentence embedding** — one vector for a whole piece of text, produced by an embedding model; used for search & RAG.

## Explanation

### What it is
Computers can't understand words, only numbers. A token ID like `3797` says nothing about meaning.
An embedding turns that ID into a **vector of meaning**, learned during training, so that:

```text
"cat"    → [ 0.21, -0.43, 0.88, …]  ┐ close together
"kitten" → [ 0.19, -0.40, 0.91, …]  ┘
"car"    → [-0.65,  0.12, 0.03, …]    far away
```

**Key idea:** *distance = difference in meaning.* Similar things → nearby vectors.

### Famous intuition
Directions in the space can encode relationships:

```text
king − man + woman ≈ queen
Paris − France + Italy ≈ Rome
```

### Two places embeddings show up
| | Inside an LLM | Embedding models (for apps) |
|---|---|---|
| **What** | One vector per token, the model's input layer | One vector per sentence/chunk/document |
| **Role** | First step before transformer layers; refined into *contextual* meaning by attention | Turn text into searchable vectors |
| **Used for** | Generating text | Semantic search, **RAG**, recommendations, clustering, classification |
| **Examples** | Part of GPT, Claude, Llama | OpenAI `text-embedding-3`, Voyage, BGE, Sentence-Transformers |

### Static vs contextual
- **Static** (Word2Vec, GloVe): one fixed vector per word — "bank" is the same in *river bank* and *bank account*.
- **Contextual** (transformers): the vector changes with surrounding words after attention — "bank" gets a different meaning in each sentence.

### Why it matters for LLM apps
- **Semantic search** — finds "how to reset password" even if the doc says "recover account login".
- **RAG** — embed document chunks → store in a **vector database** → embed the question → retrieve closest chunks → give them to the LLM.
- Pick one embedding model and stick with it: vectors from different models are **not comparable**.

## Examples

Cosine similarity between embeddings:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Sentence embeddings with `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")      # 384-dim vectors
emb = model.encode(["I love cats", "Kittens are great", "The stock market fell"])

print(emb.shape)                                      # (3, 384)
print(model.similarity(emb[0], emb[1]))               # high  — similar meaning
print(model.similarity(emb[0], emb[2]))               # low   — unrelated
```

## Open Questions
- [ ] How are embeddings actually learned during training?
- [ ] How do I choose the right embedding dimension/model for a RAG project?

## Related
- [Tokenization — Best Practices](tokenization.md) — tokens are what get embedded
- [Transformers — The Basics](transformers.md) — embeddings are the transformer's input
