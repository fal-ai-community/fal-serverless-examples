# fal-serverless-examples — assistant instructions

## Conventions

- One example = one folder under its tier dir, copied from `templates/example/`. Never create examples outside tier dirs.
- CI verifies every example by real ephemeral deploy + smoke test (`harness/check.py`) — a failing example fails the run.
- Never publish a performance claim that can't be backed by a reproducible run documented alongside it.
- Dependencies: `App.requirements` inside `app.py` is what fal installs at deploy time and is authoritative; `requirements.txt` mirrors it exactly for local venvs/CI — keep them identical, pinned. No unpinned installs in READMEs.
- Every example README follows `templates/example/README.md` (the authoritative template): problem → output shown first → run it → how it works → deploy → cost & tuning → next steps.
- One folder per catalog slot: short kebab-case name, no slot prefix (e.g. `00-getting-started/hello-fal/`); the slot ↔ folder mapping lives in the tier README table.
- Keep `llms.txt` in sync: every tier README and example is indexed there.
- Target Python: 3.11 (matches CI).
