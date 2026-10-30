# Changelog

All notable changes are listed here. The format follows Keep a Changelog; versions
follow semantic versioning.

## [Unreleased]

### Fixed

- `--require-level` and `assert_level` treated `primal_feasible` and `dual_feasible` as
  ordered; they are incomparable, and any certificate that implies the level now counts.
- A solver's non-finite objective (or overflowed float64 residuals) killed a study driver
  thread silently; such values are recorded as null and a failed driver fails the run.
- Report thresholds compare against the exact decimal eps rather than its float.
- The CLI's `--require-level` checked only the as-is and best certificates; it now checks
  every repair certificate, like `assert_level`.
- `Certificate.to_dict` raised OverflowError for objectives beyond the float range.
- The QPS reader falls back to fixed-format MPS, so QFORPLAN (names with spaces) loads:
  all 138 Maros-Meszaros files now do.
- HiGHS tolerance requests also set `kkt_tolerance` and `ipm_optimality_tolerance`.
- `study run` refuses to start when a needed solver is missing (its units became
  permanent error records); the launcher installs all extras.
- `fetch` refuses a changed unpinned file unless `--accept-new-data`, and refuses
  redirects away from the source hosts.
- `--require-eps` is validated when arguments are parsed.
- The pytest plugin is the top-level module `pytest_certkkt` and imports certkkt lazily,
  so it no longer slows down or breaks unrelated test sessions.

### Added

- Report: paired solver comparison of exact optima (exact McNemar, Holm-adjusted),
  as-is certification at each solver's default tolerance, float-versus-exact
  discrepancy size, and repair budget exhaustion counts.
- Study manifests record the SHA-256 and size of every input file; the French returns
  snapshot behind the quick study is in `docs/results/quick-french-SHA256SUMS`.

## [0.1.0] - 2026-10-30

### Added

- Exact rational problem model (zero, nonnegative and second-order cones; quadratic
  objectives) with exact MPS/QPS and CBF readers and cvxpy input.
- Exact certification of primal-dual candidates: residuals, cone membership, gap and
  complementarity, certificate levels, exact PSD check for QP dual bounds.
- Repairs: equality restoration and dual fix by weighted least-norm corrections,
  rational snapping, exact active-set KKT solves, interior push for SOCPs.
- Adapters for Clarabel, SCS, OSQP, ECOS and HiGHS with convention tests.
- CLI (`solve`, `certify`, `solvers`, `fetch`, `study`), pytest plugin and helpers.
- Checkpointed study runner over Maros-Meszaros, a CBLIB SOCP subset, the downloadable
  Mittelmann SOCP instances and a portfolio SOCP family; report with Wilson intervals.
