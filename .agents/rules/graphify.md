---
trigger: always_on
description: Always use graphify as the default codebase navigation, architecture analysis, and dependency resolution system across all projects.
---

## Graphify Default System Rules

Graphify is the DEFAULT knowledge engine for all projects and codebases.

### 1. Automatic Initialization
- Whenever starting work on any project or codebase:
  - If `graphify-out/graph.json` is missing, automatically initialize it using:
    `python -m graphify extract . --code-only && python -m graphify export html`
  - Ensure `graphify-out/` is ready before answering large architecture or dependency questions.

### 2. Primary Navigation Over Grep
- ALWAYS query the graph first instead of dumping raw source files into context:
  - Use `graphify query` for concept and subsystem searches.
  - Use `graphify path Source Target --undirected` to trace relationships and cross-module interactions.
  - Use `graphify explain Symbol` to analyze blast radius, inbound callers, and dependencies before modifying any class or function.
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files.
- Read `graphify-out/GRAPH_REPORT.md` for high-level god nodes and community architecture reviews.

### 3. Continuous Synchronization
- After modifying, creating, or refactoring code files in any session, run `python -m graphify extract . --code-only` (or `graphify update .`) to keep the knowledge graph in sync.
- In Git repositories, ensure `graphify hook install` is active so commits automatically refresh the graph.
