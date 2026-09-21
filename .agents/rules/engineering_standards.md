---
trigger: always_on
description: Project engineering rules integrating Graphify, Ponytail, and core engineering skills.
---

## 1. Graphify Knowledge Engine
- Graphify is the DEFAULT knowledge engine for this project.
- Query the graph (`graphify query`, `graphify path`, `graphify explain`) before grepping or dumping raw files.
- If `graphify-out/graph.json` is missing, auto-initialize with:
  `python -m graphify extract . --code-only && python -m graphify export html`
- After modifying code files, keep the graph in sync using `graphify update .`.

## 2. Ponytail (Minimalism & YAGNI)
- Use standard library and existing project utilities before creating new abstractions or adding dependencies.
- No boilerplate, speculative scaffolding, or single-use factories.
- The best code is the code you never wrote.

## 3. Engineering Rigor
- Follow Red-Green-Refactor with `test-driven-development` when implementing logic.
- Conduct a 5-axis review with `code-review-and-quality` before finalizing major changes.
- Trace root causes with `debugging-and-error-recovery` rather than patching symptoms.
