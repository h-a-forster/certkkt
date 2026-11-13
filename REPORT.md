# certkkt study: report (quick profile)

Date: 2026-10-30; own stopping rules (section 2.1) and the Mittelmann check (section 5)
added 2026-11-13. Profile: `quick` (DESIGN section 5.4). The full profile was started
and stopped by owner decision at 348 of 3093 units; nothing here comes from it (see
AGENTS.md). Every number below comes from
[docs/results/quick-units.jsonl](docs/results/quick-units.jsonl) (one record per unit)
and, for the own stopping rules,
[docs/results/quick-own-tests.jsonl](docs/results/quick-own-tests.jsonl).
`certkkt study report --own-tests docs/results/quick-own-tests.jsonl` generates the
tables and the certification, own-rule, pairing, float64-versus-exact and
certificate-level counts into [docs/results/quick.md](docs/results/quick.md); the
remaining figures (binding measures, gap ranges, repair diagnostics) are direct counts
over the same records.

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

Exact arithmetic earns its place here through certificates, not verdicts. The exact
repairs turned default solver output into an exact rational optimum for 29 of the 40
Maros-Meszaros problems, and HiGHS output reaches every one of the 29 (finding 1). Judged
by its own stopping rule, evaluated exactly, no solver's output fails it on any problem
where it claimed optimal (finding 2; for OSQP one condition is mostly undetermined, and
HiGHS's evaluated gate is one it enforces itself). The failure counts in certkkt's
uniform measure (finding 3) are therefore a statement about that measure, not about the
solvers keeping their promises, and they need no exact arithmetic (finding 4).

Maros-Meszaros subset at solver defaults (n = 40 problems per solver; nominal 95% Wilson
intervals, see section 6):

| solver | exact optimum after repair | claims optimal | broke its own stopping rule | misses certkkt's measure at own default tol | as-is certified at 1e-6 | as-is at 1e-8 |
|---|---|---|---|---|---|---|
| HiGHS | 29/40 = 73% [57, 84] | 39/40 | 0/39 = 0% [0, 9] | 25/39 (1e-7) | 23/40 = 58% [42, 71] | 5/40 = 13% [5, 26] |
| Clarabel | 23/40 = 58% [42, 71] | 40/40 | 0/40 = 0% [0, 9] | 10/40 (1e-8) | 37/40 = 93% [80, 97] | 30/40 = 75% [60, 86] |
| SCS | 22/40 = 55% [40, 69] | 40/40 | 0/40 = 0% [0, 9] | 10/40 (1e-4) | 16/40 = 40% [26, 55] | 7/40 = 18% [9, 32] |
| OSQP | 19/40 = 48% [33, 63] | 33/40 | 0/33 = 0% [0, 10]; 29 undetermined | 12/33 (1e-3) | 3/40 = 8% [3, 20] | 0/40 = 0% [0, 9] |

"Exact optimum" means the repairs found an exactly feasible primal-dual pair with gap
exactly zero (an exact rational optimum). "Broke its own stopping rule": the solver
claimed optimal, yet its returned output fails a condition of the test it stops on
(section 2.1), in exact arithmetic; "undetermined": the exact verdict depends on a
quantity the solver does not return. "Misses certkkt's measure": some exact relative
residual (primal, dual, gap; DESIGN 3.2) exceeds the value of the solver's documented
default tolerance; "certified at eps" means all three are at most eps.

1. **Exact optima, paired by problem.** 29/40 problems reach an exact optimum from at
   least one solver (19/40 from all four; 11/40 from none), and every one of the 29 is
   reached from HiGHS output: no problem had an exact optimum from Clarabel, SCS or OSQP
   output alone. HiGHS beats OSQP after a Holm correction over the 6 pairs (10 problems
   to 0, exact McNemar p = 0.002, adjusted 0.012); its lead over SCS (7 to 0, p = 0.016)
   and Clarabel (6 to 0, p = 0.031) is suggestive but does not survive the correction.
   All 93 exact optima (HiGHS 29, Clarabel 23, SCS 22, OSQP 19) came from the active-set
   repair (exact KKT solve on the active set read off the floats), including from poor
   floats (OSQP: 0/40 certified at 1e-8 as is, 19/40 exact optima). The exact optimum
   depends only on the problem, so these counts measure how well each solver's output
   identifies the active set; HiGHS is an active-set QP solver. None of the 243 float
   outputs is exactly feasible on both sides as returned (216 `residuals`, 27
   `primal_feasible`); only the exact repairs turn them into certificates. Beyond the
   optima they found an exactly feasible primal-dual pair bracketing the optimum
   (`bounded`) for all 72 portfolio units and 16 more Maros-Meszaros units. These counts
   depend on the 30 s repair budget and machine load (section 6).
2. **No solver's output fails its own stopping rule.** Each solver's termination test
   was re-implemented from its source and evaluated exactly on the output it returned
   (section 2.1). On no unit claiming optimal, in any suite or tolerance, does the output
   fail it: 0/40 Clarabel, 0/39 HiGHS, 0/40 SCS and 0/33 OSQP on Maros-Meszaros, and 0 of
   the 3 Clarabel and 4 SCS CBLIB units and the 24 Clarabel and 24 SCS portfolio units.
   Two qualifications. OSQP passes exactly on only 4 of its 33: on the other 29 its dual
   residual and gap pass exactly, but its primal residual is measured against an
   internal iterate it does not return, and the exact verdict depends on that iterate
   (undetermined); OSQP's own float report of that residual passes on all 29. HiGHS's
   active-set solver stops on a stationarity test on its own working set, which is not
   evaluated; what is evaluated is the bound and dual-sign gate HiGHS applies after it,
   which turns any failure into a solve error, so a returned "Optimal" passes it in
   floats by construction and 0/39 only confirms that rounding changes nothing. The
   margins are small where the solver checks often: the largest value/threshold ratio is
   0.994 (OSQP), 0.974 (SCS), 0.924 (HiGHS) and 0.853 (Clarabel) on Maros-Meszaros (for
   OSQP's primal residual, at the nearest admissible iterate). The solvers' own float
   reports of the same quantities differ from the exact values by at most 0.4% of the
   threshold, so exact arithmetic changes no verdict here either. As expected, the 5
   OSQP units that stopped on an iteration cap or claimed dual infeasibility fail its
   rule; of the 3 "inaccurate" units, Clarabel's passes the reduced one and OSQP's 2 pass
   it on dual residual and gap, primal undetermined.
3. **certkkt's uniform measure is stricter where the solvers normalise differently.**
   Among units claiming optimal, certkkt's measure at the value of each solver's default
   tolerance is missed by 10/40 Clarabel, 25/39 HiGHS, 10/40 SCS and 12/33 OSQP units.
   56 of these 57 misses bind on the dual residual or the gap, the conditions the solvers
   scale differently: certkkt divides the dual residual by `1 + ||q||` and the gap by
   `1 + |pobj| + |dobj|`; OSQP and SCS divide by the largest term (`||P x||`, `||A'y||`;
   `|x'P x|`, `|q'x|`, the dual term), Clarabel by sums of 2-norms that include `|x|` and
   `|z|`, and HiGHS's active-set solver tests stationarity only through its search
   direction on its working set (`||p||_2 < 1e-11`, a step, not a residual), on a
   Hessian regularised by 1e-7, and tests no gap; its 1e-7 tolerances apply to bound and
   dual-sign feasibility. The 2026-11-12 audit's count for OSQP holds: 35 of the
   37 OSQP units failing 1e-6 bind on the dual residual (19) or the gap (16). At 1e-6, the
   exact residuals exceed the threshold for 3/40 Clarabel, 16/39 HiGHS, 24/40 SCS and
   30/33 OSQP units claiming optimal; over all default units failing 1e-6 the binding
   measure is the dual residual (45) or the gap (32) far more often than primal
   feasibility (11, 7 of them SCS). The quick profile runs Maros-Meszaros at defaults
   only, so whether loose defaults explain the OSQP and SCS rates at 1e-6 is a hypothesis
   for a tolerance sweep to test.
4. **Float64 evaluation of the residuals was never wrong.** On all 243 units with a
   solution (every suite and tolerance), float64 and exact evaluation agreed about
   `max_rel <= eps` at 1e-4, 1e-6 and 1e-8 (largest absolute difference 2e-11). The
   threshold test alone has little power (12 units have an exact `max_rel` within a
   factor of 2 of 1e-6, 18 of 1e-8), but the discrepancies are small throughout: the
   largest |log10(float64 / exact)| of `max_rel` is 0.15, at values near 1e-16. The as-is
   columns and findings 2 and 3 need no exact arithmetic.
5. **Portfolio SOCPs: exact two-sided bounds from every solve; the gap is what differs.**
   All 72 portfolio units (every solver and tolerance) were repaired to `bounded`
   certificates (exactly feasible primal and dual). At defaults the exact relative gap is
   at most 1e-6 for 6/6 Clarabel, 6/6 ECOS and 1/6 SCS solves (at most 1e-8: 3/6, 5/6 and
   0/6); SCS default gaps range from 2.2e-7 to 6.7e-4. The interior push costs about two
   orders of magnitude of relative gap against the as-is gap: the repaired gap tracks the
   as-is `max_rel`, not the as-is `rel_gap` (Clarabel defaults: as-is `rel_gap` 8e-12 to
   5e-10, repaired gap 7e-9 to 5e-8). The optima are irrational, so `optimal` is not
   expected.
6. **CBLIB SOCPs: no exact certificate within the quick budget.** 0/12 units reached
   `bounded`; the interior repair exhausted its 30 s budget on all four instances (20-44
   thousand nonzeros). Section 4 describes the second obstacle, found on nql30.

### 2.1 How the own stopping rules were evaluated

[src/certkkt/own_tests.py](src/certkkt/own_tests.py) re-implements each solver's
termination test from its source, for the versions run (OSQP C 1.0.0 as bundled by
osqp 1.1.3, `src/auxil.c`; SCS C 3.3.1 at commit cc244eb, `src/scs.c`; Clarabel 0.11.1,
`info.rs`; HiGHS 1.15.1, `quass.cpp`, `Highs::callSolveQp`, `checkOptimality`,
`getKktFailures`), with its norms, unscaling and default tolerances (DESIGN 3.5 gives the
conditions and file citations). It evaluates them exactly on the doubles the solver was
given and the vectors it returned. Three conditions cannot be decided from the output.
OSQP's primal residual uses an internal iterate `z` it does not return, so it counts as
passing only if it passes for every `z` in `[l, u]` and as failing only if it fails for
every one; otherwise it is undetermined, with OSQP's own float report kept beside it
(31 units: 29 claiming optimal and 2 inaccurate, all passing by that report). Clarabel's
`kappa/tau <= 1` is internal to its embedding. HiGHS's active-set stop needs its working
set and regularised Hessian, so only the gate it applies afterwards is evaluated. The vectors are not in
`quick-units.jsonl`, so `certkkt study own-tests` re-solved all 216 Clarabel, HiGHS, OSQP
and SCS units with their recorded settings; all 216 reproduced the recorded status, raw
status, iteration count and objective bit for bit (once SCS's BLAS ran single-threaded,
as in the study's workers; DESIGN 8), so the evaluation describes the recorded outputs.
Records: [docs/results/quick-own-tests.jsonl](docs/results/quick-own-tests.jsonl); table:
[docs/results/quick.md](docs/results/quick.md), section "Each solver against its own
stopping rule". ECOS (24 portfolio and 4 CBLIB units) has no evaluator.

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
  equality rows exactly independent. A rank-revealing basis choice is the planned fix.

## 5. Mittelmann SOCP benchmark and the "Clarabel 14 of 18" figure

A separate one-day check (2026-11-13), not part of the quick profile's tables.

- **Where the figure comes from.** Not from Mittelmann: his "Large Second Order Cone
  Benchmark" (https://plato.asu.edu/ftp/socp.html) lists MOSEK, ECOS, KNITRO, COPT,
  Gurobi, OPTVERSE, cuOpt or COPT-PDHCG in every version checked (the live page dated
  26 Oct 2026 and Internet Archive snapshots of pages dated 12 Nov 2022, 11 Sep 2024,
  31 Oct 2024, 5 Apr 2025, 29 Oct 2025, 9 Jan 2026 and 10 Aug 2026), never Clarabel, and
  its log directory has no Clarabel folder. The figure is Table 7 of Li, Huang, Liu, Ge
  and Ye, "GPU-Accelerated Conic Quadratic Programming with Local Linear Convergence under
  Strict Complementarity" (arXiv 2608.09159v1): on "the 18 public instances in the
  Mittelmann collection", Clarabel solves 14 at target accuracy 1e-4 (shifted geometric
  mean 601.98 s) and 14 at 1e-6 (602.04 s), with a 3600 s limit on an Intel Xeon Platinum
  8469C with 512 GB RAM. The preprint gives no Clarabel version or settings and no
  per-instance results, so which 4 instances failed is not stated.
- **It does not contradict Clarabel's paper.** Goulart and Chen (arXiv 2405.12762) do not
  run this set. Their SOCP results are on PGLib-OPF relaxations (at least reduced
  accuracy on more than 99% of cases), and they write that all solvers tested "struggled
  to some extent" to solve large-scale SOCPs to full accuracy. Different instances, so
  there is nothing to reconcile.
- **All 18 instances are freely downloadable.** An earlier version of this section said
  11 were not on CBLIB; they are, with a `2013_` prefix (`2013_firL1`, `2013_dsNRL`, ...;
  74 to 914 MB compressed, 4.2 GB together). Their variable counts match Mittelmann's
  column counts (checked for firL1Linfeps, wbNRL and dsNRL). `certkkt fetch mittelmann`
  now covers all 18.
- **What ran.** Clarabel 0.11.1 at the preprint's tolerances (`tol_feas = tol_gap_abs =
  tol_gap_rel = eps`) and at its defaults (1e-8), 3600 s limit, one thread, on this laptop
  (Windows 11, about 14 GB RAM shared with other jobs), data read straight into doubles
  (`cbf.read_cbf_float`, identical to the exact reader's doubles). A run was kept under
  3 hours, so only the 8 instances up to 15M nonzeros ran, at 1e-6 only for the larger
  four. Records: [docs/results/mittelmann-clarabel.jsonl](docs/results/mittelmann-clarabel.jsonl);
  script: [scripts/mittelmann_clarabel.py](scripts/mittelmann_clarabel.py).

| instance | nonzeros | 1e-6 | 1e-4 | default (1e-8) |
|---|---|---|---|---|
| chainsing-50000-1 | 1.6M | Solved, 11 s | Solved, 8 s | Solved, 12 s |
| chainsing-50000-2 | 1.3M | Solved, 31 s | Solved, 27 s | AlmostSolved, 34 s |
| chainsing-50000-3 | 0.9M | Solved, 16 s | Solved, 15 s | AlmostSolved, 30 s |
| db-plate-yield-line | 1.6M | Solved, 25 s | Solved, 20 s | Solved, 84 s |
| db-joint-soerensen | 6.0M | NumericalError, 562 s (19 iterations) | - | - |
| 2013_firL1Linfeps | 9.9M | time limit, 3738 s (66 iterations) | - | - |
| 2013_firL2L1alph | 10.0M | Solved, 772 s (9 iterations) | - | - |
| beam7 | 15.3M | Solved, 757 s (43 iterations) | - | - |

- **Result.** At 1e-6 Clarabel solved 6 of the 8. db-joint-soerensen ends in a numerical
  error, which faster hardware would not change (KNITRO and ECOS also fail it on
  Mittelmann's page); firL1Linfeps hit the time limit after 66 iterations at about a
  minute each, which the preprint's faster machine may or may not avoid (MOSEK needs
  21 s). If both behave there as here, 2 of the preprint's 4 failures are among these 8
  and its 14/18 needs 2 failures among the 10 not run; if only the numerical error
  carries over, 3. Either is plausible, so this run neither confirms nor refutes the
  figure. Mittelmann runs codes "in default mode": at Clarabel's defaults two of the four
  smallest instances end AlmostSolved, so a default-mode count would be lower than 14.
  Where the output was small enough (up to 2M nonzeros), every unit passes Clarabel's own
  stopping rule exactly (section 2.1; reduced tolerances for the AlmostSolved ones).
- **What finishing the count needs.** The 10 instances not run: beam30 (64M nonzeros),
  2013_firL2Linfeps (19M), 2013_firL1 and 2013_firL2L1eps (40M each), 2013_wbNRL (39M),
  2013_firL2a (50M), 2013_dsNRL (67M), 2013_firL1Linfalph and 2013_firLinf (80M each),
  2013_firL2Linfalph (122M). Clarabel's working set, sampled once during each solve, was
  2.7 GB on db-joint-soerensen, 1.9 GB on firL1Linfeps and 2.2 GB on beam7, so the
  largest plausibly need 16-32 GB (an extrapolation, not measured); a
  machine with at least 32 GB, the 4.2 GB download (`certkkt fetch mittelmann`), and up to
  1 hour per instance and tolerance (up to 20 hours for both tolerances) with
  `scripts/mittelmann_clarabel.py`, which resumes and budgets its runs.

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
- certkkt's relative measures are not the solvers' stopping criteria (section 2
  compares the two); "meets the requested tolerance" is in certkkt's measure. The own
  rules are evaluated on the output the solver returned, which can differ by rounding
  from the scaled iterate it tested; the solvers' float reports differ from the exact
  values by at most 0.4% of the threshold.
- Problems are certified as published (exact decimals); the solvers saw the nearest
  doubles.
- Infeasibility and unboundedness claims are not certified. One OSQP unit reported
  PRIMALC5 dual infeasible although it is a feasible, bounded problem; the point OSQP
  returned has exact `max_rel` 5e12.
