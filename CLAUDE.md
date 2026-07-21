# fal-serverless-examples — assistant instructions

**Start every session by reading [`CONTEXT.md`](CONTEXT.md)** — it holds the full project context: what this repo is, the 40-entry catalog, all Linear ticket links, open decisions, and the build order.

## Conventions

- One example = one folder under its tier dir, copied from `templates/example/`. Never create examples outside tier dirs.
- CI verifies every example by real ephemeral deploy + smoke test (`harness/check.py`) — a failing example fails the run. Never publish performance claims that can't be backed; the measurement layer (numbers.json) was descoped 2026-07-21 (see CONTEXT.md) and lives in git history if it returns.
- Dependencies: `App.requirements` inside `app.py` is what fal installs at deploy time and is authoritative; `requirements.txt` mirrors it exactly for local venvs/CI — keep them identical, pinned. No unpinned installs in READMEs.
- Every example README follows `templates/example/README.md` (the authoritative template, adapted from design-doc §4-A): problem → output shown first → run it → how it works → deploy → cost & tuning → next steps.
- Docs pages live in the separate fal-docs repo (github.com/fal-ai/fal-docs), branched from **current origin/main** — this repo holds the runnable source those pages link to.
- The Linear ticket is the source of truth for each example's scope; its Definition of Done includes a validated deployment (older tickets say "measured numbers via the harness" — read that as the CI deploy-check per the 2026-07-21 descope in CONTEXT.md). If Linear is unreachable, stop and ask Rehan (project owner) — never guess scope.
- One folder per catalog slot: short kebab-case name, no slot prefix (e.g. `00-getting-started/hello-fal/`); the slot ↔ folder mapping lives in the tier README table.
- Target Python: 3.11 (matches CI).
