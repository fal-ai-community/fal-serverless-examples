# fal-serverless-examples — assistant instructions

## Conventions

- One example = one folder under its tier dir: short kebab-case name (e.g. `00-getting-started/hello-fal/`) containing `app.py`, a pinned `requirements.txt`, `test_app.py`, and `README.md`. Never create examples outside tier dirs.
- CI verifies every example by real ephemeral deploy + smoke test (`harness/check.py`) — a failing example fails the run. Never publish performance claims that can't be backed by a reproducible run.
- Dependencies: `App.requirements` inside `app.py` is what fal installs at deploy time and is authoritative; `requirements.txt` mirrors it exactly for local venvs/CI — keep them identical, pinned. No unpinned installs in READMEs.
- Every example README follows: problem → output shown first → run it → how it works → deploy → cost & tuning → next steps.
- The docs pages that walk through examples live in the separate fal-docs repo (github.com/fal-ai/fal-docs) — this repo holds the runnable source those pages link to.
- Update `llms.txt` when adding an example.
- Target Python: 3.11 (matches CI).
