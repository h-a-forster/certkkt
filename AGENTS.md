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
- Commits end with the project's co-author trailer. The GitHub repo is private; do not
  make it public or publish releases without the owner's go-ahead.

## Architecture map

| Path | Role |
|---|---|
| `src/certkkt/rational.py` | backend (gmpy2/Fraction), exact decimals, directed rounding, sqrt bounds, snapping |
| `src/certkkt/sparse.py` | rational CSR, modular rank profile, refinement + reconstruction, exact elimination, PSD check |
| `src/certkkt/problem.py` | conic and boxed forms, dual maps, `make_problem` |
| `src/certkkt/qps.py`, `cbf.py`, `cvxpy_io.py`, `instance.py` | readers and instances |
| `src/certkkt/certify.py` | exact certificate and levels |
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
uv run pytest -m "not slow"   # about 30 s
uv run pytest                 # adds study-runner tests that spawn workers
uv run ruff check . && uv run ruff format --check .
CERTKKT_RATIONAL=fraction uv run pytest tests/test_certify.py   # pure-Python backend
```

## How to run the study

```sh
uv run certkkt fetch mm cblib french
uv run certkkt study run --profile quick --suite mm --workers 4          # ~5 min
uv run certkkt study run --profile quick --suite portfolio --suite cblib --workers 4  # ~3 min
uv run certkkt study report --profile quick --out docs/results/quick.md --json docs/results/quick.json
```

The full profile runs for hours: `bash scripts/run_study.sh` (fetch, three-attempt
retry loop, log in `runs/longrun.log`, report in `runs/full/report.md`). Environment
variables `SUITES`, `WORKERS`, `RUN_DIR`, `MAX_UNITS` and `LOG` restrict or redirect it.

## Status and next steps

Status (2026-10-30): v0.1.0. Library, CLI, plugin and adapters done and tested (140
tests, all passing locally; the CI workflow has not run on GitHub yet); quick study run
and reported in REPORT.md. The full study was started and stopped by owner decision at
348/3093 units (all Maros-Meszaros; 28 hard timeouts at 1200 s, all on CVXQP1_L, CVXQP2_L
and CVXQP3_L) after 6.4 h, its own projection then about 50 more hours
(`runs/full/progress.log`); it is not reported.

Full-profile time estimate (3 workers), from measured units: small Maros-Meszaros units
about 6 s, 100 kB-1 MB files 2-28 s, larger files about 90 s (repairs overshoot their
60 s budget; LISWET3 94 s, AUG2DQP 88 s), EXDATA (2.3M nonzeros) 23 s load and no exact
work; CBLIB default units 60-100 s; Mittelmann units 0.5-1 h each (chainsing-50000-3:
load 11 s, Clarabel 17 s, certify 8 s, then up to 300 s of repairs; SCS may use its
3600 s limit). About 6 hours in total (range 4-9 h), the Mittelmann suite being the
least certain part. That estimate did not hold (see Status): re-estimate before a
rerun, which resumes from `runs/full`. Memory: db-joint-soerensen (6M nonzeros) needs
several GB per worker while parsed as exact rationals.

Next steps, in order:

1. Cut the full profile's run time (the large Maros-Meszaros units time out), finish it
   and fold `runs/full/report.md` into REPORT.md.
2. Mittelmann SOCP: obtain the 11 instances that are not on CBLIB's download list (fir*,
   dsNRL, wbNRL) and a float-only path for beam30 (64M nonzeros) to cover all 18.
3. Repair speed: exact elimination in pure Python dominates repair time above ~10^4
   nonzeros; a compiled sparse rational LU (e.g. via FLINT) would lift the size limit.
4. Ill-conditioned equality systems (nql/qssp in CBLIB): the exact basic correction moves
   x by O(1) or more; a float rank-revealing basis choice should fix most cases.
5. Farkas certificates for infeasibility/unboundedness claims; PSD cone (where the dual
   membership certificates of Lee, Papp and Varga become necessary).
