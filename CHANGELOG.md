# Changelog

All notable changes are listed here. The format follows Keep a Changelog; versions
follow semantic versioning.

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
