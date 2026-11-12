# certkkt

Exact optimality certificates for LP, QP and SOCP solutions.

A floating-point solver reports "optimal" when its own scaled residuals fall below its
tolerances. certkkt takes the problem data and the solver's primal-dual output, converts
both to exact rationals, and decides in exact arithmetic whether the point is feasible,
whether the dual is feasible, whether every cone membership holds and how large the
duality gap really is. When the floats are close to optimal but not exactly feasible, it
tries to repair them into an exact certificate: an exactly feasible primal-dual pair whose
objective values bracket the true optimum.

- **Problems:** MPS/QPS (exact decimals), CBF (CBLIB), cvxpy problems, or the canonical
  conic form `min 1/2 x'Px + q'x  s.t.  Ax + s = b, s in K` with zero, nonnegative and
  second-order cones.
- **Solvers:** Clarabel, SCS, OSQP, ECOS and HiGHS through their native Python APIs.
- **Arithmetic:** `gmpy2.mpq` (fallback `fractions.Fraction`). Problem files are read as
  exact decimals, so `0.1` means 1/10, not its binary rounding.
- **Output:** a certificate level, exact residuals (reported as floats rounded up, so
  they are rigorous upper bounds), the exact gap and complementarity.

Status: v0.1, alpha. The certification is sound by construction (everything reported is
recomputed exactly); the repairs are heuristics whose success rate is measured in
[REPORT.md](REPORT.md).

## Install

```sh
uv add "certkkt[all] @ git+https://github.com/h-a-forster/certkkt"   # or: pip install ...
```

Extras: `clarabel`, `scs`, `osqp`, `ecos` (wheels up to Python 3.12), `highs`, `cvxpy`,
`solvers` (all five), `all` (solvers and cvxpy). On Python 3.13 the extras skip ECOS, so
study runs of the cblib, portfolio and mittelmann suites refuse to start; the study and CI
use Python 3.12 (`.python-version`).

## Use

Command line:

```sh
certkkt solve QAFIRO.QPS --solver clarabel          # solve, certify, repair
certkkt solve nql30.cbf.gz --solver ecos --json
certkkt certify problem.qps solution.json --require-level bounded   # exit 1 if weaker
certkkt solvers
```

`solution.json` holds `{"x": [...], "z": [...]}` in the canonical convention
(`P x + q + A'z = 0`, `z` in the dual cone), or for MPS/QPS input
`{"x": [...], "y_row": [...], "y_col": [...]}` with OSQP's sign convention.

Library:

```python
from certkkt import load_instance, certify, repair
from certkkt.solvers import get_adapter

inst = load_instance("QAFIRO.QPS")
res = get_adapter("highs").solve(inst)
cert = certify(inst.problem, res.x, res.z)  # the floats, checked exactly
print(cert.level, cert.to_dict()["max_rel"])  # residuals 2.9e-05 (HiGHS 1.15.1)
best = repair(inst.problem, res.x, res.z).best
print(best.level, best.pobj - best.dobj)  # optimal 0 (active-set repair)
```

pytest (the plugin registers itself):

```python
def test_my_solver(certkkt):  # tolerance from --certkkt-eps (default 1e-6)
    certkkt.assert_certified(problem, x, z)
    certkkt.assert_level(problem, x, z, "bounded")  # repairs first
```

## Certificate levels

| level | meaning |
|---|---|
| `optimal` | primal and dual exactly feasible, gap exactly 0: an exact optimum |
| `bounded` | primal and dual exactly feasible: the optimum lies in `[dobj, pobj]` |
| `primal_feasible` | an exactly feasible point: `pobj` is an upper bound |
| `dual_feasible` | an exactly feasible dual: `dobj` is a lower bound |
| `residuals` | neither side exactly feasible; exact residuals reported |

Relative measures: `rel_primal = r_P / (1 + ||b||_inf)`, `rel_dual = r_D / (1 + ||q||_inf)`,
`rel_gap = |pobj - dobj| / (1 + |pobj| + |dobj|)`; `max_rel` is the largest of the three.
For QPs the dual bound is valid only if `P` is positive semidefinite, which certkkt
decides exactly; otherwise dual-based levels are withheld and nothing counts as certified
at an eps. Details:
[docs/DESIGN.md](docs/DESIGN.md).

## Repairs

`as_is` (the floats), `fixed` (exact equality restoration and dual fix), `snapped`
(simplest rationals, then `fixed`), `active_set` (exact KKT solve on the guessed active
set; LP/QP) and `interior` (push into the cone interiors, then a weighted least-norm
exact projection back onto the affine constraints, in the spirit of Lee, Papp and Varga;
the route to exact certificates for SOCPs, whose optima are usually irrational). Every
candidate is re-certified exactly; a repair can strengthen a certificate but never make a
false claim.

## Study

The benchmark study (Maros-Meszaros, a CBLIB SOCP subset, the downloadable Mittelmann
SOCP instances and a min-variance portfolio SOCP family built from Ken French industry
returns) is in [REPORT.md](REPORT.md); how it is run is in
[docs/DESIGN.md](docs/DESIGN.md) section 5.

```sh
certkkt fetch mm cblib french
certkkt study run --profile quick --workers 4      # resumable
certkkt study report --profile quick
```

## Related work

VIPR, exact SCIP and SoPlex certify LP and MIP results exactly; qpbenchmark compares QP
solvers by tolerance-based residuals; CVQP (arXiv:2504.10814) is an operator-splitting
method for large CVaR-constrained QPs. certkkt covers convex QP and SOCP solutions and
reports exact residuals. See [docs/DESIGN.md](docs/DESIGN.md) section 7.

## Licence

MIT. Benchmark data are downloaded from their sources, not redistributed.
