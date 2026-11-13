# certkkt 0.1.0

First release (2026-10-30; alpha, the API may change), tagged `v0.1.0` and published as a
GitHub release on 2026-11-08. Install:

```sh
uv add "certkkt[all] @ git+https://github.com/h-a-forster/certkkt"
```

## What is in it

- Exact certificates for LP, QP and SOCP solutions: problem data (MPS/QPS and CBF read
  as exact decimals, or cvxpy problems) and a solver's primal-dual floats are converted
  to rationals, and feasibility, cone membership, the duality gap and complementarity
  are decided exactly (`gmpy2.mpq`, fallback `fractions.Fraction`). Only `certify.py`
  assigns a certificate level.
- Repairs that turn near-optimal floats into exact certificates: least-norm equality
  restoration and dual fix, rational snapping, exact active-set KKT solves and an
  interior push for SOCPs. They are heuristics; a failed repair proves nothing.
- Adapters with convention tests for Clarabel 0.11.1, SCS 3.3.1, OSQP 1.1.3, ECOS 2.0.14
  and HiGHS 1.15.1; tolerance requests map to each solver's own options (DESIGN 2).
- CLI (`solve`, `certify`, `solvers`, `fetch`, `study`), a pytest plugin with a
  `certkkt` fixture, and assertion helpers.
- A checkpointed study runner (one record per unit, resume by key, hard per-unit
  timeout) over Maros-Meszaros, a CBLIB SOCP subset, the downloadable Mittelmann SOCP
  instances and a portfolio SOCP family, with a report using Wilson intervals and
  Holm-adjusted exact McNemar tests.

## Quick study (`REPORT.md`, `docs/results/`)

244 units: the 40 Maros-Meszaros QPs with QPS files of at most 100 kB x {Clarabel, OSQP,
SCS, HiGHS} at defaults, 4 CBLIB SOCPs x {Clarabel, SCS, ECOS} at defaults, and 6
portfolio SOCPs x {Clarabel, SCS, ECOS} x {default, 1e-4, 1e-6, 1e-8}. 7 min wall on 4
workers; 243 units produced a solution.

- Of the 40 QPs, default output certifies as is at 1e-6 for 37 (Clarabel), 23 (HiGHS),
  16 (SCS) and 3 (OSQP), although each solver claims optimal on 33 to 40.
- 29 of the 40 QPs reach an exact rational optimum after repair from at least one
  solver, and every one of them from HiGHS output; all 93 exact optima came from the
  active-set repair. HiGHS beats OSQP after the Holm correction (10 problems to 0,
  adjusted p = 0.012).
- Float64 evaluation of the residuals agreed with exact evaluation about `max_rel <=
  1e-8` on all 189 default units with a solution.
- All 72 portfolio units were repaired to exact two-sided bounds; 0 of 12 CBLIB units
  were, within the 30 s repair budget.

## Full study: not run

The full profile (3093 units) was started on 2026-10-30 with 3 workers and stopped by
owner decision at 348 units after 6.4 h, all of them Maros-Meszaros; 28 were hard
timeouts at 1200 s on CVXQP1_L, CVXQP2_L and CVXQP3_L, and the run's own projection
stood at about 50 more hours against an estimate of about 6. Its partial records are not
reported. So there are no results for the Maros-Meszaros and CBLIB tolerance sweeps, the
larger Maros-Meszaros problems, the other CBLIB instances, the rest of the portfolio
family or the Mittelmann instances.

## Changes after the first draft of these notes (2026-11-06)

- `--require-eps` and `assert_certified(repair_first=True)` now accept any repair
  certificate that meets eps. They looked at the top-ranked one only, so an exact pair
  with a wide gap could fail a check that the floats themselves pass.
- Nothing counts as certified at an eps when `P` is not known to be PSD: zero residuals
  on a non-convex QP only show a KKT point, and `--require-eps` used to pass on a saddle.
- `psd_check`'s budget counts the size of each pivot. With plain updates it ran past the
  1200 s hard timeout on CVXQP1_L, which accounts for the timeouts of the stopped full
  run; those problems now end with PSD status "unknown" after about a minute.
- A repair method that fails numerically (float steps overflow on huge solver output) is
  noted and dropped; `repair` used to raise and lose the as-is certificate.
- The QPS reader refuses duplicate RHS and RANGES entries, as it does in COLUMNS.
- `REPORT.md` section 3 named the wrong binding measure: on the Clarabel and ECOS
  portfolio units it is the primal cone violation, not the dual residual.
- A rerun of the quick study reproduced every status, as-is residual and table
  (`docs/results/quick-rerun.md`). `scripts/highs_tolerances.py` also prints the count of
  solutions that move with the tolerance, which DESIGN cites.

## Limits

- Infeasibility and unboundedness claims are not certified; there is no PSD cone.
- Exact elimination is pure Python and dominates repair time above about 10^4 nonzeros;
  ill-conditioned equality systems (nql30) defeat the exact correction.
- At most 7 of the 18 Mittelmann SOCP instances are publicly downloadable.
- The CI workflow has not run on GitHub; the 146 tests pass locally (Windows, Python
  3.12).

The full list of changes is in [CHANGELOG.md](../CHANGELOG.md).
