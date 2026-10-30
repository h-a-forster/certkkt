import math

import numpy as np
import pytest

from certkkt import Q, certify, make_problem, repair
from certkkt.certify import LEVELS


def lp():
    """min x1 + x2  s.t.  x1 + x2 >= 1, x >= 0  (optimal value 1)."""
    return make_problem(
        A=[[-1, -1], [-1, 0], [0, -1]], b=[-1, 0, 0], cones=[("nonneg", 3)], q=[1, 1]
    )


def disc(c=(-1, 0)):
    """min c'x  s.t.  ||x||_2 <= 1  as  s = (1, x1, x2) in SOC."""
    return make_problem(A=[[0, 0], [-1, 0], [0, -1]], b=[1, 0, 0], cones=[("soc", 3)], q=c)


def test_exact_optimum_is_optimal():
    cert = certify(lp(), [Q(1, 2), Q(1, 2)], [1, 0, 0])
    assert cert.level == "optimal"
    assert cert.gap == 0 and cert.complementarity == 0
    assert cert.certified_at("0")


def test_primal_infeasible_point_is_refuted_with_exact_residual():
    cert = certify(lp(), [Q(1, 2), Q(1, 2) - Q(1, 10**9)], [1, 0, 0])
    assert cert.level == "dual_feasible"
    assert cert.primal_cone_violation == Q(1, 10**9)
    assert not cert.certified_at("1e-10") and cert.certified_at("1e-9")


def test_floats_are_taken_at_their_exact_binary_value():
    # 0.1 + 0.9 is 1 in float64, but the binary values of 0.1 and 0.9 do not sum to 1.
    cert = certify(lp(), np.array([0.1, 0.9]), np.array([1.0, 0.0, 0.0]))
    assert 0.1 + 0.9 == 1.0
    assert cert.primal_member is (Q(0.1) + Q(0.9) >= 1)
    assert cert.primal_residual == max(Q(0), 1 - Q(0.1) - Q(0.9))


def test_soc_optimum_and_dual_cone_violation():
    p = disc()
    assert certify(p, [1, 0], [1, -1, 0]).level == "optimal"
    bad = certify(p, [1, 0], [Q(99, 100), -1, 0])  # z0 < ||z1:||
    assert not bad.dual_member
    assert bad.dual_cone_violation >= Q(1, 100)
    assert bad.dual_cone_violation - Q(1, 100) < Q(1, 10**15)


def test_dual_bound_needs_psd_objective():
    # P = diag(1, -1) is indefinite: the Wolfe dual is not a valid lower bound.
    p = make_problem(
        A=[[-1, 0], [0, -1], [1, 0], [0, 1]],
        b=[0, 0, 1, 1],
        cones=[("nonneg", 4)],
        q=[0, 0],
        P=[[1, 0], [0, -1]],
    )
    cert = certify(p, [0, 0], [0, 0, 0, 0])
    assert cert.psd == "not_psd"
    assert cert.level == "primal_feasible"
    assert any("semidefinite" in n for n in cert.notes)


def test_to_dict_handles_huge_rationals():
    # A gap whose numerator and denominator have thousands of digits used to raise
    # ValueError (Python's int-to-str digit limit) in to_dict.
    p = lp()
    big = Q(10**5000 + 1, 10**5000)
    d = certify(p, [Q(1, 2), Q(1, 2) * big], [1, 0, 0]).to_dict()
    assert d["exact_gap"] is None
    assert d["gap_up"] > 0  # rounded up, so a positive gap stays positive
    assert certify(p, [Q(1, 2), Q(1, 2)], [1, 0, 0]).to_dict()["exact_gap"] == "0"
    # Objectives beyond the float range become inf instead of raising OverflowError.
    assert certify(p, [Q(10) ** 400, 0], [1, 0, 0]).to_dict()["pobj"] == math.inf


def test_rel_measures_upper_bound_float_values():
    p = disc((-1, -1))
    x = np.array([1 / math.sqrt(2)] * 2)
    z = np.array([math.sqrt(2), -1.0, -1.0])
    d = certify(p, x, z).to_dict()
    assert d["max_rel"] < 1e-15
    assert d["level"] != "optimal"  # the optimum -sqrt(2) is irrational


def test_repair_turns_float_lp_solution_into_exact_optimum():
    out = repair(lp(), np.array([0.5 + 1e-12, 0.5 - 3e-12]), np.array([1 + 1e-13, 2e-14, -1e-14]))
    assert out.best.level == "optimal"
    assert out.best.pobj == 1


def test_repair_brackets_irrational_soc_optimum():
    p = disc((-1, -1))
    x = np.array([1 / math.sqrt(2)] * 2)
    z = np.array([math.sqrt(2), -1.0, -1.0])
    best = repair(p, x, z, stop_at_optimal=False).best
    assert best.level == "bounded"
    # The exact bracket [dobj, pobj] must contain -sqrt(2) and be tight.
    assert best.dobj < 0 and best.dobj * best.dobj >= 2
    assert best.pobj >= 0 or best.pobj * best.pobj <= 2
    assert float(best.pobj - best.dobj) < 1e-6


def test_repair_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown repair method"):
        repair(lp(), np.zeros(2), None, methods=["nope"])


def test_rank_prefers_small_residuals_over_useless_one_sided_certificates():
    p = lp()
    one_sided = certify(p, [0, 0], [1, 0, 0])  # exact dual, primal violated by 1
    near = certify(p, [Q(1, 2), Q(1, 2) - Q(1, 10**9)], [1, -Q(1, 10**12), 0])
    assert (one_sided.level, near.level) == ("dual_feasible", "residuals")
    assert near.rank() < one_sided.rank()
    assert certify(p, [Q(1, 2), Q(1, 2)], [1, 0, 0]).rank() < near.rank()


def test_levels_order():
    assert LEVELS[0] == "optimal" and LEVELS[-1] == "residuals"
