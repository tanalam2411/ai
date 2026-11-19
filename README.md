# 🧠 AI Learning Journal

My personal notes from learning AI, following
📖 **The Hitchhiker's Guide to Agentic AI: From Foundations to Systems** (Haggai Roitman, [arXiv:2606.24937](https://arxiv.org/abs/2606.24937)) — one folder per book chapter.
Maintained automatically by Claude (see [CLAUDE.md](CLAUDE.md)).

📓 **[Learning Log](LEARNING-LOG.md)** — dated session entries

## 📂 Structure

```
ai/
├── README.md          # this index
├── LEARNING-LOG.md    # dated session log
├── CLAUDE.md          # rules for maintaining the journal
└── chapter-NN/        # one folder per course chapter (chapter-01/, chapter-15/, …)
    └── <topic>/       # fundamentals/, llms/, rag/, agents/, prompting/, …
        └── NN-<concept>.md  # one note per concept, numbered in learning order
```

## 📚 Chapters, Topics & Notes

## [Chapter 1](chapter-01/)

### [LLMs](chapter-01/llms/)
_Read in order:_
1. [Tokenization — Best Practices](chapter-01/llms/01-tokenization.md) — vocab size, special tokens, fertility, numbers, code
2. [Embeddings — The Basics](chapter-01/llms/02-embeddings.md) — meaning as vectors, similarity, token vs text embeddings, RAG use
3. [Transformers — The Basics](chapter-01/llms/03-transformers.md) — what it is, role in LLMs, self-attention, variants
4. [Parameters, Hyperparameters & Sampling](chapter-01/llms/04-parameters.md) — weights & biases, hyperparameters, temperature / top-k / top-p, why it matters to users & agents
5. [Inference — The Basics](chapter-01/llms/05-inference.md) — training vs inference, generation loop, sampling, speed, cost, where it runs
6. [Decoding](chapter-01/llms/06-decoding.md) — greedy, beam search, temperature / top-k / top-p / min-p, repetition penalties, constrained decoding, self-consistency, speculative decoding
7. [Fine-Tuning & PEFT (LoRA)](chapter-01/llms/07-fine-tuning-peft.md) — full vs parameter-efficient tuning, adapters, LoRA/QLoRA, end-to-end PEFT example
8. [Knowledge Distillation](chapter-01/llms/08-distillation.md) — teacher → student, purpose, DIY recipe, use cases incl. agents

### [Prompting](chapter-01/prompting/)
_Read in order:_
1. [In-Context Learning](chapter-01/prompting/01-in-context-learning.md) — learning from prompt examples, zero/few-shot, ICL vs fine-tuning

<!-- Index format (chapters, then topics and notes in learning order):
## [Chapter N](chapter-NN/)
### [Topic](chapter-NN/topic/)
1. [Note title](chapter-NN/topic/01-note.md) — one-line summary
-->

## 🎯 Target Skills

- [ref-1](https://jobs.smartrecruiters.com/Mirantis/744000095929406-software-engineer-open-source-contributor-remote-)
  - Experience and understanding of how to leverage AI-powered co-development tools such as Cursor IDE, Claude Code, Cline, Aider.Chat or similar.
  - Foundational understanding of AI infrastructure: inferencing, Agentic workflows, autonomous AI agents, the Model Context Protocol (MCP), serverless AI functions, and related.
