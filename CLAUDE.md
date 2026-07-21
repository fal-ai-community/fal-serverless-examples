# fal-serverless-examples — assistant instructions

**Start every session by reading [`CONTEXT.md`](CONTEXT.md)** — it holds the full project context: what this repo is, the 40-entry catalog, all Linear ticket links, open decisions, and the build order.

## Conventions

- One example = one folder under its tier dir, copied from `templates/example/`. Never create examples outside tier dirs.
- `numbers.json` is written only by `harness/measure.py` — run locally or by CI, either is fine — never hand-edit or invent values; committed numbers must come from real harness runs. If a number isn't measured yet, it stays `null`.
- Dependencies: `App.requirements` inside `app.py` is what fal installs at deploy time and is authoritative; `requirements.txt` mirrors it exactly for local venvs/CI — keep them identical, pinned. No unpinned installs in READMEs.
- Every example README follows `templates/example/README.md` (the authoritative template, mirroring design-doc §4-A): problem → output shown first → run it → how it works → deploy → numbers → cost & tuning → next steps.
- Docs pages live in the separate fal-docs repo (github.com/fal-ai/fal-docs), branched from **current origin/main** — this repo holds the runnable source those pages link to.
- The Linear ticket is the source of truth for each example's scope; its Definition of Done includes a validated deployment and measured numbers. If Linear is unreachable, stop and ask Rehan (project owner) — never guess scope.
- One folder per catalog slot: short kebab-case name, no slot prefix (e.g. `00-getting-started/hello-fal/`); the slot ↔ folder mapping lives in the tier README table.
- Target Python: 3.11 (matches CI).
