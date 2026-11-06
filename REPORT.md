# certkkt study: report (quick profile)

Date: 2026-10-30. Profile: `quick` (DESIGN section 5.4). The full profile was started
and stopped by owner decision at 348 of 3093 units; nothing here comes from it (see
AGENTS.md). Every number below comes from
[docs/results/quick-units.jsonl](docs/results/quick-units.jsonl) (one record per unit) via
`certkkt study report`; the generated tables are in
[docs/results/quick.md](docs/results/quick.md).

## 1. What ran

- **Units:** 244 = Maros-Meszaros problems whose QPS file is at most 100 kB (40 of 138)
  x {Clarabel, OSQP, SCS, HiGHS} at defaults; CBLIB nql30, qssp30, sched_50_50_orig,
  sched_50_50_scaled x {Clarabel, SCS, ECOS} at defaults; 6 portfolio SOCPs (10
  industries, windows ending 2005-12, 2015-12, 2025-12, leverage 1.3 and 2) x {Clarabel,
  SCS, ECOS} x {default, 1e-4, 1e-6, 1e-8}.
- **Versions:** Python 3.12.13, gmpy2 2.3.2, numpy 2.5.4, scipy 1.18.1; Clarabel 0.11.1,
  SCS 3.3.1, OSQP 1.1.3, ECOS 2.0.14, HiGHS 1.15.1
  ([docs/results/quick-manifest.json](docs/results/quick-manifest.json)).
- **Limits:** solver 60 s, repairs 30 s per unit (all methods together), hard kill 300 s.
- **Machine and time:** Windows 11, 16 logical CPUs, 4 workers; 7 min wall in two calls.
- **Outcomes:** 243 units produced a solution and were certified; 1 had no solution
  (HiGHS "Solve error" on QSHARE1B). No timeouts or crashes.
- **Data:** problem files read as exact decimals; archives checked against recorded
  SHA-256 (`src/certkkt/datasets.py`). The French returns files have no pinned checksum
  (they change monthly upstream); the snapshot used is in
  [docs/results/quick-french-SHA256SUMS](docs/results/quick-french-SHA256SUMS).

## 2. Headline

Exact certification of default solver output, Maros-Meszaros subset (n = 40 problems per
solver; nominal 95% Wilson intervals, see section 6):

| solver | claims optimal | as-is certified at 1e-6 | as-is at 1e-8 | as-is at own default tol | exact optimum after repair |
|---|---|---|---|---|---|
| Clarabel | 40/40 | 37/40 = 93% [80, 97] | 30/40 = 75% [60, 86] | 30/40 = 75% [60, 86] (1e-8) | 23/40 = 58% [42, 71] |
| HiGHS | 39/40 | 23/40 = 58% [42, 71] | 5/40 = 13% [5, 26] | 14/40 = 35% [22, 50] (1e-7) | 29/40 = 73% [57, 84] |
| SCS | 40/40 | 16/40 = 40% [26, 55] | 7/40 = 18% [9, 32] | 30/40 = 75% [60, 86] (1e-4) | 22/40 = 55% [40, 69] |
| OSQP | 33/40 | 3/40 = 8% [3, 20] | 0/40 = 0% [0, 9] | 21/40 = 53% [37, 67] (1e-3) | 19/40 = 48% [33, 63] |

"Certified at eps" means every exact relative residual (primal, dual, gap; DESIGN 3.2)
is at most eps. "Own default tol" applies that test at the value of the solver's
documented default tolerance, in certkkt's measure. It is a yardstick, not the solver's
stopping criterion, so a miss does not by itself mean the solver broke its tolerance.
"Exact optimum" means the repairs found an exactly feasible primal-dual pair with gap
exactly zero (an exact rational optimum).

1. **Status "optimal" says little at tight thresholds.** Among units whose solver
   reported optimal, the exact residuals exceed 1e-6 for 3/40 Clarabel, 16/39 HiGHS,
   24/40 SCS and 30/33 OSQP units. Even at the value of each solver's default tolerance,
   10/40 Clarabel (1e-8), 25/39 HiGHS (1e-7), 10/40 SCS (1e-4) and 12/33 OSQP (1e-3) of
   these units exceed it in certkkt's measure. Loose defaults plausibly account for much
   of the OSQP and SCS gap at 1e-6, but the quick profile runs Maros-Meszaros at defaults
   only, so this is a hypothesis for a tolerance sweep to test. A loose default does not explain
   HiGHS, whose default feasibility tolerances are 1e-7: its QP output exceeds 1e-7 in
   certkkt's measure on 25/39 units claiming optimal, with the gap (13) or the dual
   residual (12) as the binding measure and the primal residual at most 2e-11. This does
   not show HiGHS missing its own tolerance: 1e-7 is tested in HiGHS's measures, not
   certkkt's, and the options that set a gap target change nothing on these QPs (DESIGN
   2). Over all default units failing 1e-6, the binding measure is the dual residual (45) or the gap (32) far
   more often than primal feasibility (11, 7 of them SCS).
2. **Exact optima, paired by problem.** 29/40 problems reach an exact optimum from at
   least one solver (19/40 from all four; 11/40 from none), and every one of the 29 is
   reached from HiGHS output: no problem had an exact optimum from Clarabel, SCS or OSQP
   output alone. HiGHS beats OSQP after a Holm correction over the 6 pairs (10 problems
   to 0, exact McNemar p = 0.002, adjusted 0.012); its lead over SCS (7 to 0, p = 0.016)
   and Clarabel (6 to 0, p = 0.031) is suggestive but does not survive the correction.
   All 93 exact optima came from the active-set repair (exact KKT solve on the active set
   read off the floats), including from poor floats (OSQP: 0/40 certified at 1e-8 as is,
   19/40 exact optima). The exact optimum depends only on the problem, so these counts
   measure how well each solver's output identifies the active set; HiGHS is an
   active-set QP solver.
3. **Float64 evaluation of the residuals was never wrong at 1e-8.** On all 189 default
   units with a solution, float64 and exact evaluation agreed about `max_rel <= 1e-8`.
   The threshold test alone has little power (only 14 units have an exact `max_rel`
   within a factor of 2 of 1e-8), but the discrepancies are small throughout: the largest
   |log10(float64 / exact)| of `max_rel` is 0.15, at values near 1e-16. The value of
   exact arithmetic here lies in the certificates (exact feasibility, exact bounds, exact
   optima), not in re-evaluating residuals.
4. **Portfolio SOCPs: exact two-sided bounds from every solve; the gap is what differs.**
   All 72 portfolio units (every solver and tolerance) were repaired to `bounded`
   certificates (exactly feasible primal and dual). At defaults the exact relative gap is
   at most 1e-6 for 6/6 Clarabel, 6/6 ECOS and 1/6 SCS solves (at most 1e-8: 3/6, 5/6 and
   0/6); SCS default gaps range from 2.2e-7 to 6.7e-4. The interior push costs about two
   orders of magnitude of relative gap against the as-is gap: the repaired gap tracks the
   as-is `max_rel`, not the as-is `rel_gap` (Clarabel defaults: as-is `rel_gap` 8e-12 to
   5e-10, repaired gap 7e-9 to 5e-8). The optima are irrational, so `optimal` is not
   expected.
5. **CBLIB SOCPs: no exact certificate within the quick budget.** 0/12 units reached
   `bounded`; the interior repair exhausted its 30 s budget on all four instances (20-44
   thousand nonzeros). Section 4 describes the second obstacle, found on nql30.

## 3. Requested tolerance versus certified accuracy (portfolio family)

Median [quartiles] of log10 of the exact as-is `max_rel`, n = 6 per cell; in brackets
after the cell, how many of 6 meet the requested tolerance in certkkt's measure:

| solver | default | tol 1e-4 | tol 1e-6 | tol 1e-8 |
|---|---|---|---|---|
| Clarabel | -7.6 [-7.9, -7.4] | -3.6 [-3.8, -3.4] (0/6) | -6.1 [-6.2, -5.7] (4/6) | -7.6 [-7.9, -7.4] (1/6) |
| ECOS | -7.5 [-7.6, -7.3] | -3.6 [-3.9, -3.4] (2/6) | -5.4 [-5.7, -5.3] (1/6) | -7.5 [-7.6, -7.3] (0/6) |
| SCS | -6.1 [-6.6, -4.7] | -6.1 [-6.6, -4.7] (6/6) | -7.1 [-9.7, -6.6] (5/6) | -9.2 [-10.1, -8.6] (5/6) |

- The default columns equal the solvers' documented defaults: Clarabel and ECOS default
  to 1e-8, SCS to 1e-4 (the records are identical), which checks the tolerance mapping.
- The interior-point solvers land within about half an order of magnitude of the
  request, mostly above it, so a request of 1e-8 is met in certkkt's measure for 1/6
  (Clarabel) and 0/6 (ECOS) units. SCS lands one to two orders below its request, so its
  requested tolerance is conservative in this measure.
- The exact relative gap is two to three orders smaller than `max_rel` in the median of
  every cell. The binding measure is the primal cone violation of `s = b - A x` on all 48
  Clarabel and ECOS units; on SCS it is the dual residual (17 of 24 units) or the primal
  cone violation (7).

## 4. Repair diagnostics

- **QPs:** of the 109 mm units with an `optimal` or `bounded` best certificate, 93 came
  from `active_set`, 7 from `snapped`, 6 from `interior` and 3 from `fixed`. Repairs used
  982 of the 990 wall-clock seconds spent on mm units (per-unit times summed over 4
  parallel workers). Much of that is budget exhaustion rather than useful work: on 15 mm
  units the interior repair ran out of the 30 s budget, and on 23 the active-set KKT
  systems were inconsistent.
- **SOCPs, why the interior push needed two changes** (both regression-tested): a cone
  whose head is a constant (the turnover budget `||w - w0|| <= tau`) or pinned by
  stationarity (the epigraph variable `t` in `||R w|| <= t`) cannot gain room by raising
  its head, so the push also shrinks tails and repeats; before that change no portfolio
  solution reached `bounded`.
- **nql30 (CBLIB):** given more time, the interior push succeeds in floating point (room
  2.6e-7 after the push), but the exact equality restoration moves x by O(10^4): the
  float least-norm step stalls at an equality residual of about 1.5e-14, and the exact
  basic correction of that remainder is badly conditioned. The instance contains
  coefficients 0.5000000000015 next to 0.5, which plausibly makes nearly dependent
  equality rows exactly independent. A rank-revealing basis choice is the planned fix
  ([backlog.md](backlog.md)).

## 5. Mittelmann SOCP benchmark: coverage and reproduction

- The benchmark has no Clarabel results to compare with. Mittelmann's "Large Second
  Order Cone Benchmark" (https://plato.asu.edu/ftp/socp.html) lists MOSEK, ECOS, KNITRO,
  COPT, Gurobi, OPTVERSE, cuOpt or COPT-PDHCG in every version checked: the live page
  dated 26 Oct 2026 and Internet Archive snapshots of pages dated 12 Nov 2022, 11 Sep
  2024, 31 Oct 2024, 5 Apr 2025, 29 Oct 2025, 9 Jan 2026 and 10 Aug 2026. None lists
  Clarabel, and the log directory (plato.asu.edu/ftp/socp_logs/) has no Clarabel folder.
- Only 7 of the 18 instances are on CBLIB's public download list (beam7, beam30,
  chainsing-50000-1/2/3, db-joint-soerensen, db-plate-yield-line); fir*, dsNRL and wbNRL
  are not. A reproduction can cover at most 7/18. The full profile runs Clarabel, SCS
  and ECOS at defaults on the 5 that fit in memory as exact rationals (beam7 and beam30,
  55 and 229 MB compressed, are recorded as skipped) with a 3600 s solver limit; it has
  not been run yet.
- Single pilot measurement (not part of the tables): Clarabel 0.11.1 at defaults on
  chainsing-50000-3 (350k variables, 950k nonzeros) returned status AlmostSolved
  ("inaccurate") after 17 s; certkkt loaded the instance in 11 s and certified the floats
  in 8 s, exact as-is `max_rel` 2.8e-9 (level `residuals`).

## 6. Limitations

- Quick profile only: small Maros-Meszaros problems (40 of 138, the easy end), 4 of 18
  CBLIB instances, 6 of 54 portfolio instances. Shares will move on the full sets.
- The counts describe this instance set exactly. The Wilson intervals are nominal: they
  would apply to a hypothetical exchangeable population of small Maros-Meszaros-like
  QPs, but the instances were selected by file size (the easy end) and families share
  data (the 6 portfolio instances use the same returns), so the intervals are optimistic
  and say nothing about the full benchmark sets.
- Repair success rates are bounded by the time budget (30 s) and by pure-Python exact
  elimination; a failed repair is not evidence that no exact certificate exists. The
  budget also makes them depend on machine load: a rerun reproduced every status,
  as-is residual and table, but the budget ran out on 7 mm units instead of 15
  ([docs/results/quick-rerun.md](docs/results/quick-rerun.md)).
- certkkt's relative measures are not the solvers' stopping criteria; "meets the
  requested tolerance" is in certkkt's measure.
- Problems are certified as published (exact decimals); the solvers saw the nearest
  doubles.
- Infeasibility and unboundedness claims are not certified. One OSQP unit reported
  PRIMALC5 dual infeasible although it is a feasible, bounded problem; the point OSQP
  returned has exact `max_rel` 5e12.
