# Backlog

## In progress

## Waiting on me

- CI passes on GitHub through one manual dispatch (it has never run, and private-repo minutes are metered).
- The 34 commits up to `e5eefeb` no longer carry a Co-Authored-By trailer, by a history rewrite and force push, if wanted before going public.
- The repository is public with v0.1.0 tagged and private vulnerability reporting enabled (SECURITY.md relies on it).

## Decided

- 2026-10-30: the full study was stopped at 348 of 3093 units; v0.1.0 reports the quick profile only.

## Proposed

- The full profile runs in affordable time (HiGHS hits its 600 s limit on the CONT, BOYD and AUG2D problems), and its report is folded into REPORT.md.
- Large PSD objectives get an exact proof without elimination (verified Cholesky or similar), so CVXQP1/2/3_L, STCQP1/2 and EXDATA can reach dual-based levels.
- All 18 Mittelmann SOCP instances are covered: the 11 not on CBLIB's download list (fir*, dsNRL, wbNRL) and a float-only path for beam30 (64M nonzeros).
- Repairs scale past about 10^4 nonzeros through a compiled sparse rational LU (for example FLINT).
- Ill-conditioned equality systems (nql and qssp in CBLIB) repair with a small correction, through a float rank-revealing basis choice.
- Infeasibility and unboundedness claims are certified by Farkas rays, and the PSD cone is supported (with the dual membership certificates of Lee, Papp and Varga).
- The full report states whether HiGHS solved a slightly different problem on KSIP, HUESTIS and HUES-MOD, where its file reader drops matrix entries of size at most 1e-9.
- Study workers really run single-threaded BLAS: `_worker` sets `OMP_NUM_THREADS` after numpy is already imported.
