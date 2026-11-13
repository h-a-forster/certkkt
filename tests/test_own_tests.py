"""Each solver's own stopping rule, evaluated exactly.

The oracle is the solver itself: on a run it declares converged, the exact rule must
pass and agree with the solver's own float report of the same residuals; on a run it
stops early (iteration cap), the exact rule must fail.
"""

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp
from test_solvers import boxed_qp, conic_qp, socp

from certkkt import make_problem
from certkkt.instance import Instance
from certkkt.own_tests import OwnTestUnavailable, own_test
from certkkt.report import SELF_REPORT
from certkkt.solvers import get_adapter
from certkkt.solvers.base import NativeRun, SolverResult

pytestmark = pytest.mark.solver

RULED = ("clarabel", "highs", "osqp", "scs")
#: settings that stop each solver after too few iterations to converge
EARLY = {
    "clarabel": {"max_iter": 2},
    "osqp": {"max_iter": 5},
    "scs": {"max_iters": 5, "acceleration_lookback": 0},
}


def _solve(name, inst, settings=None):
    a = get_adapter(name)
    if not a.available():
        pytest.skip(f"{name} not installed")
    if a.unsupported_reason(inst):
        pytest.skip(a.unsupported_reason(inst))
    return a.solve(inst, settings or {})


@pytest.mark.parametrize("name", RULED)
@pytest.mark.parametrize("make", [boxed_qp, conic_qp, socp], ids=["boxed", "conic", "socp"])
def test_converged_runs_pass_their_own_rule_and_match_the_solver_report(name, make):
    res = _solve(name, make())
    assert res.status == "optimal"
    t = own_test(res)
    assert t.passes is True, t.to_dict()
    for c in t.criteria:
        key = SELF_REPORT[name].get(c.name)
        if key is None:
            continue
        reported = abs(t.solver_report[key])
        # The solver evaluates in floats from its scaled iterate; the exact value of the
        # returned output may differ by rounding, not by a factor.
        assert abs(reported - c.value) <= 1e-6 * c.threshold + 1e-9 * reported, (c, reported)


@pytest.mark.parametrize("name", sorted(EARLY))
def test_runs_stopped_early_fail_their_own_rule(name):
    res = _solve(name, boxed_qp() if name == "osqp" else socp(), EARLY[name])
    assert res.status in ("max_iter", "inaccurate"), res.raw_status
    if res.status == "inaccurate":  # the reduced rule applies; check the full one fails
        res = dataclasses.replace(res, status="optimal")
    assert own_test(res).passes is False


def _with_output(res: SolverResult, **output) -> SolverResult:
    nat = dataclasses.replace(res.native, output={**res.native.output, **output})
    return dataclasses.replace(res, native=nat)


def test_highs_gate_is_bound_and_sign_feasibility_only():
    res = _solve("highs", boxed_qp())
    out = res.native.output
    lo, up = res.native.data["col_lower"], res.native.data["col_upper"]
    j = int(np.flatnonzero(np.isfinite(lo))[0])
    # Stationarity is not part of HiGHS's QP gate: a dual off by 1 with the right sign
    # for its side of the bound interval passes ...
    below_middle = not np.isfinite(up[j]) or out["col_value"][j] < (lo[j] + up[j]) / 2
    dual = out["col_dual"].copy()
    dual[j] += 1.0 if below_middle else -1.0
    assert own_test(_with_output(res, col_dual=dual)).passes is True
    # ... but a bound violation beyond 1e-7 fails it.
    x = out["col_value"].copy()
    x[j] = lo[j] - 2e-7
    t = own_test(_with_output(res, col_value=x))
    assert t.passes is False and t.binding == "primal feasibility"


def test_osqp_primal_rule_fails_only_when_no_internal_iterate_could_pass():
    # min x^2 s.t. 0 <= x <= 1 (one row): OSQP's z is not returned, only z in [0, 1].
    nat = NativeRun(
        form="box",
        data={
            "P": sp.csc_matrix([[2.0]]),
            "q": np.zeros(1),
            "A": sp.csc_matrix([[1.0]]),
            "l": np.zeros(1),
            "u": np.ones(1),
            "infinity": 1e30,
        },
        output={"x": np.array([1.0005]), "y": np.zeros(1)},
        settings={"eps_abs": 1e-3, "eps_rel": 1e-3, "scaling": 10, "check_dualgap": 0},
    )

    def verdict(x: float, y: float = 0.0) -> bool | None:
        out = {"x": np.array([x]), "y": np.array([y])}
        r = SolverResult("osqp", "", "optimal", "", x=out["x"], z=out["y"])
        r.native = dataclasses.replace(nat, output=out)
        return next(c for c in own_test(r).criteria if c.name == "primal residual").passes

    assert verdict(1.0005) is True  # 5e-4 < 1e-3 + 1e-3 * 1.0005
    assert verdict(1.01) is False  # 1e-2 >= every threshold any z in [0, 1] allows


def test_rule_is_unavailable_rather_than_guessed():
    res = _solve("osqp", boxed_qp(), {"polishing": True})
    with pytest.raises(OwnTestUnavailable):
        own_test(res)
    ecos = _solve("ecos", socp())
    with pytest.raises(OwnTestUnavailable):
        own_test(ecos)


def test_clarabel_presolve_rows_are_left_out():
    # x <= 1 and x <= inf (nonneg rows); Clarabel drops the infinite row before it stops.
    p = make_problem(A=[[1], [1]], b=[1, 0], cones=[("nonneg", 2)], q=[-1])
    inst = Instance(problem=p)
    inst.float_data().b[1] = np.inf  # the adapters read this cached copy
    res = _solve("clarabel", inst)
    t = own_test(res)
    assert res.status == "optimal" and t.passes is True
    assert t.diagnostics["presolve_removed_rows"] == 1
