# graphify-enrich

Re-applies this project's context enrichment on top of a graphify graph.

## Why this exists

`graphify` builds `graphify-out/graph.json` from AST only. Its nodes carry a
label, a source file and a line number — nothing that says what a symbol *is*,
and nothing connecting the project's markdown to the code it describes.

This script adds both, deterministically, with no LLM and no API key:

1. **`context` on every node** — pulled from the source itself: Python
   docstrings, C/C++ preceding comment blocks, declarations and signatures, file
   headers. Symbols with no definition in the repo (`ndarray`, `MPI_Comm`) are
   described by which files reference them.
2. **`doc_context`** — the project's markdown prose duplicated into the nodes it
   describes, plus one document node per markdown file and `documents` edges
   into the code.

## Run it

```bash
python .claude/graphify-enrich/enrich.py
graphify export html      # refresh the interactive view
```

**Run it after every graphify rebuild** — `graphify update .`, `/graphify`, or
the post-commit hook. Those regenerate `graph.json` from scratch and drop
everything above.

Idempotent: it purges its own prior output before rebuilding, so repeated runs
converge on the same graph. Verified stable across consecutive runs.

Flags: `--only context` / `--only markdown` to run one stage, `--no-report` to
skip regenerating `GRAPH_REPORT.md`.

## How markdown is matched to code

Each markdown section is resolved by the first rule that hits. Every attachment
records which rule matched it in its `match` field, so a precise reference is
never mistaken for a scope-level association:

| Rule | Meaning | Confidence |
|---|---|---|
| `symbol` | the section names an identifier or path that resolves to a node | EXTRACTED |
| `inherited` | no symbol of its own; takes its parent heading's matches | INFERRED |
| `dir-scope` | code files living in the markdown's own directory | INFERRED |
| `token-scope` | filename tokens (`SIMULACIONES_RECONNECTION` → `*reconnection*`) | INFERRED |
| `doc-self` | no code referent exists; anchored to its own document node | INFERRED |

`doc-self` is deliberate. Three markdown files in this repo — `CLAUDE.md`,
`memory/projects/graz-relocation.md`, `presentation/README.md` — document the
project or its logistics, not code. They become document nodes carrying their
full text and are **not** given invented edges into the codebase.

## Invariants worth preserving

The document node for a markdown file retains **every** section of that file.
Code nodes keep only their most relevant few (`MAX_DOC_PER_NODE`), ranked by
rule precision. Without the document node holding the complete set, that cap
silently drops sections out of the graph entirely.

The heading-inheritance stack resets per file. Carrying it across documents lets
a section inherit matches from an unrelated markdown — invented edges.
