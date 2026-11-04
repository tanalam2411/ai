# CLAUDE.md — AI Learning Journal

This repo is the owner's personal AI learning journal. Claude manages it end to end:
the owner only shares what they learned; Claude writes, organizes, commits and pushes.
Never ask the owner to do anything manually.

## Workflow (every time the owner shares a learning)

1. **Write a note** — turn it into a clean, concise Markdown note (template below).
2. **File it by topic** — save under a topic folder (e.g. `fundamentals/`, `llms/`, `rag/`,
   `agents/`, `prompting/`). Create a new folder when a new topic appears.
   - File names: `kebab-case.md` (e.g. `llms/tokenization.md`).
   - **No duplicates:** if a note on the same concept exists, update it (append new insights,
     bump the `Updated` date) instead of creating a new file.
3. **Log the session** — add a dated entry to `LEARNING-LOG.md` (newest first):
   what I learned, what confused me, what's next. One entry per day; extend it if it exists.
4. **Update the index** — keep `README.md` linking every topic folder and note.
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
