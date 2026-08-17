# PSC — Anisotropic Plasma Simulations

Master's thesis codebase: PSC (Particle-in-Cell) runs and analysis of temperature-anisotropy
instabilities (mirror / firehose / whistler), comparing bi-Maxwellian and bi-kappa velocity
distribution functions.

## Layout

| Path | What it is |
|------|------------|
| `src/` | PSC source (upstream code plus the case files for this thesis) |
| `CodeforAnalisys/` | Python analysis pipeline — see its README for the full workflow |
| `cosma_jobs/` | SLURM job scripts for the COSMA HPC cluster |
| `python/` | Notebooks and helper scripts |

## Terms

| Term | Meaning |
|------|---------|
| **PSC** | The Particle-in-Cell code in this repository |
| **PIC** | Particle-in-Cell |
| **Mirror / Firehose / Whistler** | Temperature-anisotropy instability families studied here |
| **bi-kappa / bi-Maxwellian** | Non-Maxwellian vs Maxwellian velocity distributions compared across cases |
| **COSMA** | HPC cluster where production runs are executed |
| **VDF** | Velocity Distribution Function |

## Conventions

- All documentation, figure labels and deliverables are written in **English**, so that
  collaborators outside the group can follow the work.
- Figure styling goes through `CodeforAnalisys/plot_style.py`; the `paper` theme
  (white background, 300 dpi, PDF) is the default. Override with `PSC_FIG_THEME`.

## Data policy

Simulation output and analysis products are **not** tracked in git:

- `analysis_results/` — figures, GIFs and per-step maps. Fully regenerable from the raw
  run data via the `CodeforAnalisys/Makefile` targets.
- Raw run data (`*.bp`, `*.h5`, checkpoints) lives on COSMA, never in the repository.

Only source code, job scripts and documentation belong here. Keep it that way — the
history is already large.

## graphify

This project has a knowledge graph at `graphify-out/` (generated, untracked) with god nodes,
community structure and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json`
  exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"`
  for focused concepts. These return a scoped subgraph, usually much smaller than
  `GRAPH_REPORT.md` or raw grep output.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review, or when
  query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- **Always re-run `python .claude/graphify-enrich/enrich.py` after any graphify rebuild**
  (`graphify update .`, `/graphify`, or the post-commit hook). A rebuild regenerates
  `graph.json` from bare AST and drops the per-node `context` and the markdown `doc_context`
  this project depends on. The script is deterministic and idempotent — no LLM, no API key.
  See `.claude/graphify-enrich/README.md`.
