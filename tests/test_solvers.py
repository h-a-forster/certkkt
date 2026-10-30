"""Every installed adapter returns duals in the canonical convention.

A sign or ordering error in an adapter shows up here as a large exact dual residual.
"""

import numpy as np
import pytest

from certkkt import certify, make_problem, repair
from certkkt.instance import Instance
from certkkt.qps import parse_qps
from certkkt.solvers import ADAPTERS, get_adapter

pytestmark = pytest.mark.solver

QP = """\
NAME          QP2
ROWS
 N  obj
 G  c1
 L  c2
 E  c3
COLUMNS
    x1        obj       -1           c1        1
    x1        c2        1            c3        1
    x2        obj       -3           c1        1
    x2        c3        -1
    x3        c2        2            obj       0.5
RHS
    rhs       c1        0.5          c2        3
    rhs       c3        -0.25
RANGES
    rng       c2        4
BOUNDS
 UP bnd       x1        2
 LO bnd       x2        -1
 UP bnd       x2        1.5
 MI bnd       x3
QUADOBJ
    x1        x1        2
    x2        x2        1
    x1        x2        0.5
    x3        x3        1
ENDATA
"""


def boxed_qp() -> Instance:
    return Instance.from_boxed(parse_qps(QP))


def conic_qp() -> Instance:
    return Instance(problem=boxed_qp().problem)  # same problem, no boxed form


def socp() -> Instance:
    """min -x1 - 2 x2 + x3  s.t.  ||(x1, x2)|| <= x3 - 1/2, x3 <= 3, x1 + x2 = 1."""
    p = make_problem(
        A=[[1, 1, 0], [0, 0, 1], [0, 0, -1], [-1, 0, 0], [0, -1, 0]],
        b=[1, 3, "-0.5", 0, 0],
        cones=[("zero", 1), ("nonneg", 1), ("soc", 3)],
        q=[-1, -2, 1],
    )
    return Instance(problem=p)


def _available(name):
    a = get_adapter(name)
    if not a.available():
        pytest.skip(f"{name} not installed")
    return a


@pytest.mark.parametrize("name", sorted(ADAPTERS))
@pytest.mark.parametrize("make", [boxed_qp, conic_qp, socp], ids=["boxed", "conic", "socp"])
def test_adapter_convention(name, make):
    a = _available(name)
    inst = make()
    res = a.solve(inst)
    reason = a.unsupported_reason(inst)
    if reason:
        assert res.status == "unsupported"
        return
    assert res.status == "optimal", (res.raw_status, res.message)
    cert = certify(inst.problem, res.x, res.z)
    # Loose: solver defaults differ, but a convention error gives residuals of order 1.
    assert cert.certified_at("1e-3"), cert.to_dict()
    best = repair(inst.problem, res.x, res.z).best
    assert best.level in ("optimal", "bounded"), best.to_dict()


def test_tolerance_and_time_limit_settings_are_accepted():
    for name in sorted(ADAPTERS):
        a = get_adapter(name)
        if not a.available():
            continue
        inst = boxed_qp() if a.unsupported_reason(socp()) else socp()
        res = a.solve(inst, {**a.tolerance_settings(1e-9), **a.time_limit_settings(30)})
        assert res.status == "optimal", (name, res.raw_status, res.message)


def test_unknown_setting_is_an_error_not_ignored():
    a = _available("clarabel")
    res = a.solve(socp(), {"no_such_setting": 1})
    assert res.status == "error" and "no_such_setting" in res.message


def test_unknown_solver():
    with pytest.raises(ValueError, match="unknown solver"):
        get_adapter("gurobi")


def test_cvxpy_problem_round_trip():
    cp = pytest.importorskip("cvxpy")
    from certkkt.cvxpy_io import from_cvxpy

    x = cp.Variable(3)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(x) - x[0] + 2),
        [cp.norm(x[1:], 2) <= x[0], cp.sum(x) == 1, x[0] <= 0.8],
    )
    prob.solve(solver=cp.CLARABEL)
    inst = from_cvxpy(prob)
    res = _available("clarabel").solve(inst)
    assert res.status == "optimal"
    assert np.isclose(res.objective, prob.value, rtol=1e-6, atol=1e-7)
    cert = certify(inst.problem, res.x, res.z)
    assert cert.certified_at("1e-6")


@pytest.mark.parametrize("name", ["clarabel", "ecos"])
def test_interior_repair_certifies_a_portfolio_socp(name):
    # Both cones are active at the optimum and the turnover cone's head is a constant,
    # so the interior push must shrink tails, not only raise heads.
    from test_study import french_csv

    from certkkt.portfolio import min_variance_socp, parse_french_csv

    a = _available(name)
    window = parse_french_csv(french_csv()).window(202512, 60)
    inst = Instance(problem=min_variance_socp(window, "1.3", "0.05"))
    res = a.solve(inst)
    assert res.status == "optimal"
    best = repair(inst.problem, res.x, res.z).best
    assert best.level == "bounded", best.to_dict()
    assert best.max_rel < 1e-6
