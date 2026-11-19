# CLAUDE.md — AI Learning Journal

This repo is the owner's personal AI learning journal. Claude manages it end to end:
the owner only shares what they learned; Claude writes, organizes, commits and pushes.
Never ask the owner to do anything manually.

## Source book

The journal follows **The Hitchhiker's Guide to Agentic AI: From Foundations to Systems**
(Haggai Roitman, arXiv:2606.24937). Five parts: LLM foundations (architecture & optimization,
training/inference hardware & systems, classical RL) → … → agentic AI (agentic training, RAG &
agentic RAG, memory, tool integration, multi-agent orchestration, agent communication,
harness & loop engineering, agentic UIs).

- `chapter-NN/` = the book's chapter NN. The owner only names a topic; Claude decides the
  chapter and topic folder from the book's structure (current chapter by default).
- Notes from the book use `**Source:** Hitchhiker's Guide to Agentic AI, Ch. NN` (add the
  section, e.g. `§15.2`, when known).
- In `README.md`, give each chapter heading the book's chapter title once known.

## Workflow (every time the owner shares a learning)

1. **Write a note** — turn it into a clean, concise Markdown note (template below).
2. **File it by chapter, then topic** — save under the current chapter folder
   (`chapter-NN/`, e.g. `chapter-01/`, `chapter-15/`), then a topic folder inside it
   (e.g. `fundamentals/`, `llms/`, `rag/`, `agents/`, `prompting/`). Create new folders as
   needed. Chapters are kept separate: a note goes in the chapter the owner is currently
   studying (the latest chapter unless the owner names another).
   - Current chapter: **chapter-15** (chapter-01 holds all earlier notes).
   - File names: `NN-kebab-case.md`, numbered in **learning order** within each chapter/topic
     (e.g. `chapter-01/llms/01-tokenization.md`, `chapter-01/llms/02-embeddings.md`).
     When a new note belongs in the middle, insert it and renumber the following notes
     with `git mv`, then fix every link to them.
   - Chapter folders are zero-padded (`chapter-01`); topic folders stay unnumbered.
     `README.md` lists chapters in order, and topics within each chapter in learning order
     (fundamentals → llms → prompting → rag → agents …).
   - **No duplicates:** if a note on the same concept exists in the same chapter, update it (append new insights,
     bump the `Updated` date) instead of creating a new file.
3. **Log the session** — add a dated entry to `LEARNING-LOG.md` (newest first):
   what I learned, what confused me, what's next. One entry per day; extend it if it exists.
4. **Update the index** — keep `README.md` linking every chapter, topic folder and note, in learning order.
5. **Commit & push** — clear message (e.g. `notes(rag): add chunking strategies`) and push to
   `main` without asking. If pushing to `main` isn't possible, push a branch, open a pull
   request, and tell the owner the PR link.

## Note template

```markdown
# <Concept Title>

> **Date:** YYYY-MM-DD · **Updated:** YYYY-MM-DD · **Source:** <course / PDF / article, if given>

## Key Concepts
- **Term** — one-line definition

## Explanation
Short explanation framed in the owner's learning context.

## Examples
Code or worked examples, only if relevant.

## Open Questions
- [ ] Things still unclear or worth exploring

## Related
- [Other note](../topic/other-note.md)
```

## Style

- Concise and visually clean: short sections, bullets, tables where helpful, no filler.
- Omit empty sections (e.g. no Examples if none apply); leave out Source if not mentioned.
- Use relative links between notes. Use fenced code blocks with a language tag.
- Preserve the owner's wording for confusions and questions — they're part of the journal.
