# Contributing

Bug reports, new solver adapters, new cones and corrections are welcome. Read
[docs/DESIGN.md](docs/DESIGN.md) first; it fixes the conic convention, the certificate
levels and the study estimands that every change must respect.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git.

```sh
uv sync --all-extras
uv run certkkt solvers
```

## Checks

```sh
uv run pytest -m "not slow"     # fast suite
uv run pytest                   # adds the study runner tests (spawn workers)
uv run ruff check .
uv run ruff format --check .
```

CI runs the full suite on Ubuntu (Python 3.12) for pull requests and on demand.

## Rules

- Soundness first: nothing may be reported as certified unless it was checked in exact
  rational arithmetic on the exact problem data. Repairs may only propose candidates.
- Adapters convert solver output to the canonical convention and nothing else (no
  clipping, projection or rescaling). Add a convention test to `tests/test_solvers.py`.
- Every bug fix comes with a test that fails without it.
- Tests use no network; use `tmp_path` for files; they must pass on Windows.
- Dependency bumps are their own change; keep `uv.lock` committed.
