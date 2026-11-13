# AGENTS.md

Guide for agents and contributors continuing this project. The spec is
[docs/DESIGN.md](docs/DESIGN.md); read it before changing a convention, a certificate
rule, a repair or an estimand, and update it in the same change.

## What this is

A library, CLI and pytest plugin that certify LP/QP/SOCP solutions in exact rational
arithmetic and repair near-optimal floats into exact certificates, plus a CPU study of
how often default solver output (Clarabel, SCS, OSQP, ECOS, HiGHS) is certifiable.

## Conventions

- uv + hatchling, `src/certkkt`, Python 3.11+ (3.12 is the study and CI environment,
  because ECOS has no 3.13 wheel: on 3.13 cblib, portfolio and mittelmann study runs
  refuse to start). Typed code, ruff (line length 100), pytest.
- `uv.lock` is committed; `exclude-newer` keeps resolution to releases at least a week
  old. A dependency bump is its own change and invalidates published study numbers.
- Soundness rule: only `certify.py` decides levels, always from exact data and an exact
  candidate. Repairs and adapters never make claims.
- Network only in `certkkt fetch` (three hosts, SHA-256 checks). Never commit `data/`,
  `runs/` or `tmp/`; small derived results go to `docs/results/`.
- Long computations: one fsynced JSON line per unit, resume by key, `progress.log`,
  hard per-unit timeout, retry wrapper. Run anything over ~8 minutes through the
  launcher pattern (see "How to run the study").
- The GitHub repo is public; publish releases only with the owner's go-ahead. Open work
  lives in `backlog.md` (gitignored, local only).

## Architecture map

| Path | Role |
|---|---|
| `src/certkkt/rational.py` | backend (gmpy2/Fraction), exact decimals, directed rounding, sqrt bounds, snapping |
| `src/certkkt/sparse.py` | rational CSR, modular rank profile, refinement + reconstruction, exact elimination, PSD check |
| `src/certkkt/problem.py` | conic and boxed forms, dual maps, `make_problem` |
| `src/certkkt/qps.py`, `cbf.py`, `cvxpy_io.py`, `instance.py` | readers and instances |
| `src/certkkt/certify.py` | exact certificate and levels |
| `src/certkkt/own_tests.py` | each solver's own stopping rule, re-implemented from its source, evaluated exactly (DESIGN 3.5) |
| `src/certkkt/repair.py` | repair methods (DESIGN 4) |
| `src/certkkt/solvers/` | adapters and the adapter contract |
| `src/certkkt/testing.py`, `src/pytest_certkkt.py` | assertion helpers; `certkkt` fixture (top-level so pytest loads it without importing certkkt) |
| `src/certkkt/datasets.py`, `portfolio.py` | data sources and checksums; portfolio family |
| `src/certkkt/study.py`, `report.py`, `stats.py` | runner, report, Wilson intervals |
| `src/certkkt/cli.py` | CLI |
| `docs/results/` | committed study outputs (quick profile) |

## How to test

```sh
uv sync --all-extras
uv run pytest -m "not slow"   # about 10 s
uv run pytest                 # adds study-runner tests that spawn workers
uv run ruff check . && uv run ruff format --check .
CERTKKT_RATIONAL=fraction uv run pytest tests/test_certify.py   # pure-Python backend
```

## How to run the study

```sh
uv run certkkt fetch mm cblib french
uv run certkkt study run --profile quick --suite mm --workers 4          # ~5 min
uv run certkkt study run --profile quick --suite portfolio --suite cblib --workers 4  # ~3 min
uv run certkkt study own-tests --units docs/results/quick-units.jsonl --out-dir runs/quick-own  # ~3 min
uv run certkkt study report --profile quick --units docs/results/quick-units.jsonl \
    --own-tests docs/results/quick-own-tests.jsonl --out docs/results/quick.md --json docs/results/quick.json
```

`study own-tests` re-solves recorded units to evaluate each solver's own stopping rule
(DESIGN 3.5); it keeps a verdict only where the re-solve reproduces the record bit for bit,
which needs single-threaded BLAS (it sets `OMP_NUM_THREADS=1`; DESIGN 8). New study runs
record `own_test` per unit directly. `scripts/mittelmann_clarabel.py` runs Clarabel on the
Mittelmann SOCP instances through the float-only CBF reader (REPORT section 5).

Do not launch the full profile as designed (see Status). `bash scripts/run_study.sh` is
its launcher (fetch, three-attempt retry loop, log in `runs/longrun.log`, report in
`runs/full/report.md`); environment variables `SUITES`, `WORKERS`, `RUN_DIR`, `MAX_UNITS`
and `LOG` restrict or redirect it, which is the supported use until the redesign.

## Status

Status (2026-11-13): v0.1.0 plus own stopping rules. Library, CLI, plugin and adapters
done and tested; quick study run, reported in REPORT.md and reproduced by a rerun
(`docs/results/quick-rerun.md`); each solver's own stopping rule evaluated exactly on all
216 Clarabel, HiGHS, OSQP and SCS quick units (`docs/results/quick-own-tests.jsonl`; no
unit claiming optimal fails its own rule; OSQP's primal condition is undetermined on 29 of
its 33, DESIGN 3.5). The full
study was started and stopped by owner decision at 348/3093 units (all Maros-Meszaros; 28
hard timeouts at 1200 s, all on CVXQP1_L, CVXQP2_L and CVXQP3_L) after 6.4 h, its own
projection then about 50 more hours (`runs/full/progress.log`); it is not reported.

The 3093-unit full profile is not runnable as designed, and finishing it is not the plan:
it needs a redesign (size cap and repair budget, see `backlog.md`), not a
resume. Those timeouts came from `psd_check`, whose budget then counted plain updates
while the exact coefficients grew. It now counts words and ends "unknown" in about a
minute (DESIGN 3.3). That removes about half of the run's worker time (28 x 1200 s of
about 19 worker-hours), so the 50-hour figure is stale, but no new figure has been
measured and the rest of the case stands: the run had reached only 23 of the 138
Maros-Meszaros problems, and the costs below remain. On the large instances the budgets
would stop the units, so their results would measure the budget, not the solvers.
`runs/full` predates the fix and any rerun should start from an empty run directory.

Full-profile cost, from measured units: small Maros-Meszaros units about 6 s, 100 kB-1 MB
files 2-28 s, larger files about 90 s (repairs overshoot their 60 s budget; LISWET3 94 s,
AUG2DQP 88 s), but HiGHS spends its whole 600 s limit and more on the CONT, BOYD and
AUG2D problems (up to 779 s); EXDATA (2.3M nonzeros) 23 s load and no exact work; CBLIB
default units 60-100 s; Mittelmann units 0.5-1 h each (chainsing-50000-3: load 11 s,
Clarabel 17 s, certify 8 s, then up to 300 s of repairs; SCS may use its 3600 s limit).
The earlier estimate of about 6 hours did not hold. Memory:
db-joint-soerensen (6M nonzeros) needs several GB per worker while parsed as exact
rationals.
