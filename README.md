# 🧠 AI Learning Journal

My personal notes from learning AI — courses, articles and PDFs.
Maintained automatically by Claude (see [CLAUDE.md](CLAUDE.md)).

📓 **[Learning Log](LEARNING-LOG.md)** — dated session entries

## 📂 Structure

```
ai/
├── README.md          # this index
├── LEARNING-LOG.md    # dated session log
├── CLAUDE.md          # rules for maintaining the journal
└── <topic>/           # fundamentals/, llms/, rag/, agents/, prompting/, …
    └── NN-<concept>.md  # one note per concept, numbered in learning order
```

## 📚 Topics & Notes

### [LLMs](llms/)
_Read in order:_
1. [Tokenization — Best Practices](llms/01-tokenization.md) — vocab size, special tokens, fertility, numbers, code
2. [Embeddings — The Basics](llms/02-embeddings.md) — meaning as vectors, similarity, token vs text embeddings, RAG use
3. [Transformers — The Basics](llms/03-transformers.md) — what it is, role in LLMs, self-attention, variants
4. [Parameters, Hyperparameters & Sampling](llms/04-parameters.md) — weights & biases, hyperparameters, temperature / top-k / top-p, why it matters to users & agents
5. [Inference — The Basics](llms/05-inference.md) — training vs inference, generation loop, sampling, speed, cost, where it runs
6. [Fine-Tuning & PEFT (LoRA)](llms/06-fine-tuning-peft.md) — full vs parameter-efficient tuning, adapters, LoRA/QLoRA, end-to-end PEFT example

### [Prompting](prompting/)
_Read in order:_
1. [In-Context Learning](prompting/01-in-context-learning.md) — learning from prompt examples, zero/few-shot, ICL vs fine-tuning

<!-- Index format (topics and notes in learning order):
### [Topic](topic/)
1. [Note title](topic/01-note.md) — one-line summary
-->

## 🎯 Target Skills

- [ref-1](https://jobs.smartrecruiters.com/Mirantis/744000095929406-software-engineer-open-source-contributor-remote-)
  - Experience and understanding of how to leverage AI-powered co-development tools such as Cursor IDE, Claude Code, Cline, Aider.Chat or similar.
  - Foundational understanding of AI infrastructure: inferencing, Agentic workflows, autonomous AI agents, the Model Context Protocol (MCP), serverless AI functions, and related.
