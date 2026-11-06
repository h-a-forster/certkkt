# Quick study rerun, 2026-11-06

`certkkt study run --profile quick --suite mm --suite cblib --suite portfolio --workers 4`
at commit bada8e5 on the same machine, with the versions of `quick-manifest.json` and the
input files of `quick-french-SHA256SUMS`: 5 min 46 s wall in one call. Compared record by
record with `quick-units.jsonl`:

- All 244 units have the same outcome and solver status, and the as-is certificates of
  the 243 units with a solution are identical.
- The best certificate differs on 2 units, both QBRANDY, whose interior repair finished
  inside the 30 s budget this time: Clarabel moved from `residuals` to `dual_feasible`,
  and OSQP stayed at `residuals` with a smaller `max_rel`. Neither reaches `bounded`.
- `certkkt study report` on the rerun differs from `quick.md` in one line: the repair
  time budget ran out on 7 mm units instead of 15. Every table is the same.
