# certkkt design

Version 0.1 (2026-10-30). This document fixes the conventions, the certificate
semantics, the repair methods and the study estimands. Code that changes any of them
must update this file in the same change.

## 1. Goal

Given a convex LP, QP or SOCP and a floating-point primal-dual pair from a solver, say in
exact arithmetic what the pair proves:

1. **Certify or refute** primal feasibility, dual feasibility, cone membership, the
   duality gap and complementarity, with exact (or rigorously bounded) residuals.
2. **Repair** a near-optimal pair into an exact certificate when possible: an exactly
   feasible primal-dual pair, whose objective values bracket the optimum (`bounded`), or
   with zero gap (`optimal`).
3. **Measure**, on public benchmark sets, how often default solver output is certifiable
   at given exact thresholds, and how the requested tolerance maps to certified accuracy.

Non-goals for v0.1: PSD, exponential and power cones; integer variables; infeasibility
certificates (Farkas rays); proving that a solver's *status* is right when no certificate
is found.

## 2. Problem model and conventions

### 2.1 Canonical conic form (`problem.py`)

```
minimize    1/2 x'Px + q'x + r
subject to  A x + s = b,    s in K = K_1 x ... x K_p
```

Cones: `zero` (s = 0), `nonneg` (s >= 0), `soc` (s_0 >= ||s_1:||_2). Dual:

```
maximize    -1/2 x'Px - b'z + r
subject to  P x + q + A'z = 0,    z in K*
```

(the Wolfe dual; `zero* = free`, the other cones are self-dual). This is Clarabel's and
SCS's convention. All data are exact rationals; `P` is stored with both triangles and
must be symmetric. A maximisation is stored negated with `objective_sign = -1`.

### 2.2 Boxed form (MPS/QPS)

`min 1/2 x'Px + c'x + c0  s.t.  rl <= A x <= ru,  l <= x <= u`. `BoxedQP.to_conic` emits
equality rows (rows and columns with equal bounds) into the zero cone and each finite
inequality side as a nonneg row (`-a x + s = -lo`, `a x + s = up`), recording a
`RowOrigin` per conic row. Boxed multipliers (OSQP convention: stationarity
`P x + c + A'y_row + y_col = 0`, positive where an upper bound is active) map to conic
duals by `boxed_duals_to_conic`: a two-sided row splits `y` by sign; a one-sided row
passes `y` unclipped, so a wrong-signed multiplier shows up as a dual cone violation.

### 2.3 Readers

- **MPS/QPS** (`qps.py`): free format; numbers are exact decimals (`"0.1"` is 1/10).
  NAME, OBJSENSE, ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUADOBJ/QSECTION (one triangle),
  QMATRIX (full), ENDATA. Conventions as in HiGHS and CPLEX: first N row is the
  objective; an RHS on it is the negated constant; RANGES on E rows widen by sign; a
  negative UP bound with default lower bound sets the lower bound to -inf (warned).
  Integer markers and bound types are rejected.
- **CBF** (`cbf.py`): versions 1-3, cones F, L+, L-, L=, Q, QR on variables and
  constraints. A rotated cone `2 x0 x1 >= ||w||^2` maps to the SOC by the exact rational
  map `(x0 + x1/2, x0 - x1/2, w)`. PSD, exponential/power cones and integers are rejected.
- **cvxpy** (`cvxpy_io.py`): the data cvxpy compiles for Clarabel are taken as the exact
  problem (its floats convert exactly), so certificates refer to what the solver saw.

`exact_decimals=False` rounds each number to the nearest double first, so the certified
problem is exactly the one a float solver receives. The study uses exact decimals (the
problem as published), see section 8.

### 2.4 Solver adapters (`solvers/`)

Native Python APIs at solver defaults plus explicitly passed settings. Adapters change
only the convention, never the values:

| solver | input | dual mapping |
|---|---|---|
| Clarabel 0.11.1 | conic form as is | `z` as returned |
| SCS 3.3.1 | rows reordered zero, nonneg, SOC | `y` permuted back |
| ECOS 2.0.14 | zero rows as `A x = b`, rest as `G x + s = h` | `[y; z]` in row order |
| OSQP 1.1.3 | boxed form for MPS/QPS, else conic rows as `l <= A x <= u` | boxed map (2.2) |
| HiGHS 1.15.1 | boxed form for MPS/QPS, else conic rows | `y = -row_dual`, `-col_dual`, then 2.2 |

`tests/test_solvers.py` checks every adapter's convention by exact certification on a
boxed QP, the same QP in conic form and an SOCP. Tolerance requests (section 5.4) map to
Clarabel `tol_gap_abs = tol_gap_rel = tol_feas`; SCS and OSQP `eps_abs = eps_rel`; ECOS
`feastol = abstol = reltol`; HiGHS `primal_ = dual_feasibility_tolerance`. Unknown settings
are errors, not ignored.

## 3. Certification (`certify.py`)

### 3.1 Exact quantities

For a rational candidate `(x, z)` the slack is recomputed, `s = b - A x` (never taken from
the solver). Then, exactly:

- primal: equality residual `max |s_i|` over zero rows; cone violation `max(0, -s_i)` on
  nonneg rows and `||s_1:|| - s_0` on violated SOC blocks;
- dual: stationarity residual `||P x + q + A'z||_inf`; dual cone violation likewise;
- `pobj = 1/2 x'Px + q'x + r`, `dobj = -1/2 x'Px - b'z + r`, gap `pobj - dobj`,
  complementarity `s'z`.

SOC membership is decided without square roots (`s_0 >= 0` and `s_0^2 >= ||s_1:||^2`).
Violation *magnitudes* involve a square root and are reported as rational upper bounds
(`sqrt_up`, relative slack below about 2^-60).

### 3.2 Relative measures

`rel_primal = r_P / (1 + ||b||_inf)`, `rel_dual = r_D / (1 + ||q||_inf)`,
`rel_gap = |gap| / (1 + |pobj| + |dobj|)`, `max_rel = max` of the three. A candidate is
*certified at eps* when `max_rel <= eps`, compared exactly (eps is an exact decimal).
These are certkkt's measures; solvers use their own scalings, so "requested tolerance"
and "certified accuracy" are not expected to coincide (section 5.4).

### 3.3 Levels

| level | condition | claim |
|---|---|---|
| `optimal` | s in K, z in K*, stationarity exact, gap = 0 | exact optimum |
| `bounded` | both sides exactly feasible, gap > 0 | optimum in `[dobj, pobj]` |
| `primal_feasible` | s in K exactly | `pobj` is an upper bound |
| `dual_feasible` | z in K* and stationarity exact | `dobj` is a lower bound |
| `residuals` | neither | exact residuals only |

The dual bound needs `P` positive semidefinite. `psd_check` decides it exactly
(diagonal dominance test, then symmetric elimination with an operation budget); if `P` is
not PSD or the budget runs out ("unknown"), dual-based levels are downgraded and a note
says why. A negative gap for an exactly feasible pair is impossible and raises.

The levels are partially ordered: `optimal` implies `bounded`, which implies both one-sided
levels, but `primal_feasible` and `dual_feasible` are incomparable. A required level
(`--require-level`, `assert_level`) is met by any certificate that implies it.

### 3.4 Reporting

`Certificate.to_dict` rounds residuals and relative measures *up* to floats (rigorous
upper bounds), objectives to nearest, and keeps the exact gap as a string when short.
`rank()` orders certificates: `optimal`, then `bounded`, then all others by `max_rel`
(a one-sided certificate with a huge gap says less than a near-feasible pair).

## 4. Repairs (`repair.py`)

A repair proposes a rational candidate; `certify` decides. Nothing a repair computes is
trusted, so repairs affect only the success rate, never soundness. Methods:

- `as_is`: the solver floats, converted exactly.
- `fixed`: exact restoration of the zero rows and exact dual fix. The affine correction
  is the weighted least-norm step `d = D R' (R D R')^+ r` (weights: room of the tightest
  cone row a variable touches; free rows weight 1), computed by a float sparse LU of
  `R D R'` refined against the exact residual, plus an exact basic solution (modular rank
  profile, iterative refinement with rational reconstruction, exact elimination as
  fallback) for the tiny remainder. The dual fix first projects onto K*, absorbs
  stationarity residuals into multipliers of singleton (bound) rows in the manner of the
  safe bounds of Neumaier and Shcherbina, then applies the same weighted correction.
- `snapped`: each coordinate replaced by the simplest rational within 1e-12 or 1e-9
  (relative), then `fixed`; recovers optima with small denominators.
- `active_set` (LP/QP): guess the active set (multiplier exceeds slack; or slack below
  1e-7 relative) and solve the KKT system restricted to it exactly. Success gives gap 0.
- `interior`: push the slack and the dual into the interiors of their cones by relative
  margins eta in {1e-9, 1e-7, 1e-5} (SOC blocks short of room get both a raised head and a
  shrunk tail, repeated over a few rounds; equalities weighted 1e6 in the push; the dual
  margin is at least ten times the current stationarity residual), then apply the exact
  weighted corrections. This is the route to `bounded` certificates for SOCPs, whose
  optima are usually irrational. It rests on the observation of Lee, Papp and Varga
  (arXiv:2506.12256, "Dual certificates of primal cone membership") that points in the
  interior of a cone carry full-dimensional sets of easily verified membership
  certificates, so a numerical point pushed off the boundary survives exact rounding. For
  the zero, nonneg and SOC cones membership is checked directly, so their dual
  certificates of membership are not needed; extending certkkt to the PSD cone would use
  them.

Default order: `as_is, active_set, fixed, snapped, interior` for polyhedral problems and
`as_is, interior, fixed, snapped` for SOCPs, stopping at the first `optimal`. One time
budget covers all methods; a method that exceeds it or the elimination operation budget
is noted and dropped (the deadline is checked between steps, so overshoot of a few
seconds is possible).

## 5. Study (`study.py`, `datasets.py`, `portfolio.py`, `report.py`)

### 5.1 Suites and data

| suite | instances | source | integrity |
|---|---|---|---|
| `mm` | Maros-Meszaros convex QP set, 138 problems in QPDATA1-3.ZIP | https://www.doc.ic.ac.uk/~im/ | SHA-256 recorded |
| `cblib` | the 18 continuous SOCPs of the DIMACS set in CBLIB: nb, nb_L1, nb_L2, nb_L2_bessel, nql30/60/180, qssp30/60/180, sched_{50_50,100_50,100_100,200_100}_{orig,scaled} | https://cblib.zib.de/download/all/ | SHA-256 recorded |
| `mittelmann` | the 18 instances of Mittelmann's "Large Second Order Cone Benchmark" (https://plato.asu.edu/ftp/socp.html); 7 are on CBLIB's public download list (beam7, beam30, chainsing-50000-1/2/3, db-joint-soerensen, db-plate-yield-line); fir*, dsNRL and wbNRL are not | CBLIB | SHA-256 recorded for 5; beam7/beam30 recorded on first download |
| `portfolio` | 54 min-variance SOCPs (section 5.3) | Ken French data library, 10/30/49 industry portfolios | checksum recorded on first download (the files are updated monthly) |

Downloads go to `data/raw/` (gitignored); nothing is redistributed. A file whose
checksum was only recorded on first download is refused if it later differs, unless
`certkkt fetch --accept-new-data` is given, and every study call records the SHA-256 and
size of each input file in its manifest entry (`inputs`).

### 5.2 Solvers per suite

QP suites: Clarabel, OSQP, SCS, HiGHS (ECOS has no quadratic objective). SOCP suites:
Clarabel, SCS, ECOS (OSQP and HiGHS have no SOC). Versions are pinned by `uv.lock` and
recorded per call in each run's `manifest.json`.

### 5.3 Portfolio family

`K` industries, 60-month window ending `end`, gross leverage `L`, L2 turnover budget
`tau = 0.25` around equal weights:
`min t  s.t. ||R w|| <= t, sum w = 1, -u <= w <= u, sum u <= L, ||w - 1/K|| <= tau`, with
`R` the centred returns (exact two-decimal percentages; centring divides by 60).
`K in {10, 30, 49}`, `end in {1985-12, 1990-12, ..., 2025-12}`, `L in {1.3, 2}`: 54
instances. Both cones are typically active at the optimum, the turnover cone with a
constant head.

### 5.4 Units, settings, limits

A unit is `(suite, instance, solver, tol)`. `tol = default` uses the solver's defaults;
`tol in {1e-4, 1e-6, 1e-8}` requests that accuracy through the mapping of section 2.4.
Every unit also passes the solver's wall-clock limit where it has one (not ECOS); this
does not change tolerances.

| profile | mm | cblib | portfolio | mittelmann |
|---|---|---|---|---|
| quick | files <= 100 kB (40 problems), default | nql30, qssp30, sched_50_50_orig/scaled, default | 6 instances (K = 10; 2005, 2015, 2025; both L), all four tol | - |
| full | all 138, all four tol | all 18, all four tol | all 54, all four tol | 7, default |

Limits per unit (quick / full / mittelmann): solver 60 s / 600 s / 3600 s; repair budget
30 s / 60 s / 300 s; hard kill 300 s / 1200 s / 7200 s. In the full profile the
tolerance-sweep units of `mm` and `cblib` are certified as is without repairs (E3 needs
only the as-is residuals; repairs would multiply the run time by four). Exact work is
skipped above 2M (8M for mittelmann) nonzeros, and Mittelmann files above 30 MB
compressed (beam7, beam30) are not parsed (64M nonzeros do not fit in memory as Python rationals).

### 5.5 Execution

Units grouped by instance run in persistent spawn workers (one instance load per group);
the parent waits at most the hard limit per unit, then kills the worker, records
`timeout` (or `crash`) and resends the remaining units to a fresh worker. Each finished
unit is one fsynced JSON line in `runs/<profile>/units.jsonl`; reruns skip finished keys
and drop a torn last line; progress goes to `progress.log`. The long run is wrapped in a
three-attempt retry loop (the launcher script records its own log).

## 6. Estimands and reporting

Per suite and solver (and tolerance), over all planned units (failed solves, timeouts
and errors count as not certified):

- **E1 as-is certification share** at eps in {1e-4, 1e-6, 1e-8}: share of units whose
  solver floats have `max_rel <= eps` in exact arithmetic; also at each solver's
  documented default tolerance (in certkkt's measure, not the solver's).
- **E2 repaired certification share** at eps: share with a best certificate of level
  `optimal` or `bounded` and exact relative gap `<= eps` (an exactly feasible pair that
  proves eps-optimality); and **E2'** the share with an exact optimum (`optimal`).
- **E3 tolerance map**: by requested tol, the median and quartiles of log10 of the exact
  as-is `max_rel` and `rel_gap`, and the share with `max_rel <= tol`.
- **E4 float-versus-exact disagreement**: among units where both can be evaluated, those
  where float64 evaluation of the same residuals and exact evaluation disagree about
  `max_rel <= 1e-8`, with the largest log10 ratio of the two `max_rel` values.
- **E5 Mittelmann**: per instance, the solver's status, time, as-is `max_rel` and best
  level; the count of default-setting Clarabel solves with status `optimal`.
- **E6 paired solver comparison**: on the instances every solver of a suite ran, per
  solver pair, the problems with an exact optimum from only one of the two, an exact
  McNemar test and Holm adjustment over the pairs; plus the union and intersection.

The counts are exact, descriptive facts about this instance set (each solver ran once per
unit). Shares also carry 95% Wilson score intervals, which are nominal: they describe
uncertainty about a hypothetical exchangeable population of similar problems (for the
quick profile, small Maros-Meszaros-like QPs). The instances are not a random sample of
any population: they are selected by file size (the easy end) and come in families
sharing data, so the intervals are optimistic and do not extend to the full benchmark
sets.

## 7. Prior art

- **VIPR** (Cheung, Gleixner, Steffy, IPCO 2017; https://github.com/ambros-gleixner/VIPR):
  a certificate format and exact checker for MIP results. LP/MIP only.
- **Exact SCIP / SoPlex** (Eifler and Gleixner, "A computational status update for exact
  rational mixed integer programming", Math. Programming 2023; Gleixner, Steffy, Wolter,
  "Iterative refinement for linear programming", INFORMS J. Computing 2016;
  https://scipopt.org): exact rational LP/MIP solving. certkkt's `solve_refined` uses the
  same refinement plus rational reconstruction idea for its exact linear solves.
- **Vorc** (CP 2026, "VIPR Certificate Construction from Black-Box ILP Solvers";
  https://github.com/szeider/vorc): builds VIPR certificates around black-box ILP solvers.
  LP/MIP only.
- **Safe bounds** (Neumaier and Shcherbina, "Safe bounds in linear and mixed-integer
  linear programming", Math. Programming 2004): rigorous dual bounds from approximate
  duals; the singleton-row absorption in `fix_dual` follows it.
- **qpbenchmark** (https://github.com/qpsolvers/qpbenchmark): QP solver benchmark on
  Maros-Meszaros and others with tolerance-based success criteria evaluated in floating
  point. certkkt evaluates the same kind of residuals exactly.
- **CVQP** (Luxenberg, Perez-Pineiro, Diamond, Boyd, arXiv:2504.10814): an operator
  splitting method for large CVaR-constrained QPs; first-order methods of this kind stop
  at modest tolerances, which is where exact a-posteriori certificates are informative.
- **Lee, Papp, Varga** (arXiv:2506.12256): dual certificates of primal cone membership
  (section 4).
- **Mittelmann's SOCP benchmark** (https://plato.asu.edu/ftp/socp.html) and **CBLIB**
  (Friberg, Math. Programming Computation 2016; https://cblib.zib.de).
- The `lpbench` project's verification layer (residuals evaluated independently of the
  solver, an evaluation floor, "claimed optimal but failed verification" treated as a
  tolerance outcome) informed the reporting rules here; certkkt replaces float
  evaluation by exact evaluation.

## 8. Decisions and known limitations

- Exact decimals: the certified problem is the published one. A solver sees the doubles
  nearest to it, so a solution can be exactly feasible for the doubles and not for the
  decimals; the difference is below 1e-16 relative per coefficient and shows up only at
  the tightest thresholds.
- Repairs are pure Python. Exact elimination dominates repair time on problems with
  more than about 10^4 nonzeros; the time budget caps it and the report states the
  resulting success rates rather than hiding failures.
- `psd_check` and exact elimination have operation budgets; "unknown" PSD status
  downgrades dual-based certificates rather than guessing.
- Infeasible or unbounded claims are not certified (no Farkas certificates in v0.1).
