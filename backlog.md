# Backlog

## In progress

## Waiting on me

- CI passes on GitHub through one manual dispatch (it has never run, and private-repo minutes are metered).
- The 34 commits up to `bada8e5` no longer carry a Co-Authored-By trailer, by a history rewrite and force push, if wanted before going public.
- The repository is public with v0.1.0 tagged and private vulnerability reporting enabled (SECURITY.md relies on it).

## Decided

- 2026-10-30: the full study was stopped at 348 of 3093 units; v0.1.0 reports the quick profile only.
- 2026-11-07: the 3093-unit full profile will not be resumed or finished as designed; it needs the redesign under Proposed.
- 2026-11-07: rejected, review C3 (Wilson intervals need a "nominal" label): the sentence above the REPORT.md headline table already says nominal and points to section 6.
- 2026-11-07: rejected, review C1's suggested wording that HiGHS's feasibility tolerance does not bound the dual residual: HiGHS has a 1e-7 dual feasibility tolerance and its docs do not define the measure, so REPORT.md says only that certkkt's measure differs.
- 2026-11-07: rejected, review C2's claim that large instances "will not finish in any budget": repair budgets and hard kills end every unit, so the objection is cost and uninformative results.

## Proposed

- A redesigned large profile replaces the full profile: units capped by nnz (e.g. at most 2e5, dropping the unit, not only its exact work, since HiGHS alone spends 600 s on the CONT, BOYD and AUG2D problems), a repair budget tied to size, a measured estimate first, and results reported as "all Maros-Meszaros up to size X" and folded into REPORT.md.
- Large PSD objectives get an exact proof without elimination (verified Cholesky or similar), so CVXQP1/2/3_L, STCQP1/2 and EXDATA can reach dual-based levels.
- All 18 Mittelmann SOCP instances are covered: the 11 not on CBLIB's download list (fir*, dsNRL, wbNRL) and a float-only path for beam30 (64M nonzeros).
- Repairs scale past about 10^4 nonzeros through a compiled sparse rational LU (for example FLINT).
- Ill-conditioned equality systems (nql and qssp in CBLIB) repair with a small correction, through a float rank-revealing basis choice.
- Infeasibility and unboundedness claims are certified by Farkas rays, and the PSD cone is supported (with the dual membership certificates of Lee, Papp and Varga).
- The full report states whether HiGHS solved a slightly different problem on KSIP, HUESTIS and HUES-MOD, where its file reader drops matrix entries of size at most 1e-9.
- The "exact optimum after repair" column of the REPORT.md headline table carries a footnote that the counts depend on the 30 s budget and machine load (review C4).
- OSQP's "dual infeasible" on the feasible, bounded PRIMALC5 has a `docs/upstream/` note with a reproduction (review C5).
- Study workers really run single-threaded BLAS: `_worker` sets `OMP_NUM_THREADS` after numpy is already imported.
