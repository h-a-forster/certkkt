import numpy as np
import pytest

from certkkt import Q, make_problem
from certkkt.testing import assert_certified, assert_level, assert_primal_feasible


def lp():
    return make_problem(
        A=[[-1, -1], [-1, 0], [0, -1]], b=[-1, 0, 0], cones=[("nonneg", 3)], q=[1, 1]
    )


def test_assertions_pass_and_fail():
    p = lp()
    assert_certified(p, [0.5, 0.5], [1, 0, 0], eps="0")
    with pytest.raises(AssertionError):
        assert_certified(p, [0.5, 0.5 - 1e-7], [1, 0, 0], eps="1e-8")
    with pytest.raises(AssertionError):
        assert_certified(p, [0.5, 0.5], None)
    assert_primal_feasible(p, [Q(1), Q(0)], eps="0")
    with pytest.raises(AssertionError):
        assert_primal_feasible(p, [0.4, 0.5], eps="1e-3")


def test_assert_level_repairs_first():
    p = lp()
    x, z = np.array([0.5 + 1e-12, 0.5 - 1e-12]), np.array([1.0, 1e-15, 0.0])
    assert assert_level(p, x, z, "optimal").level == "optimal"
    with pytest.raises(AssertionError):
        assert_level(p, x, z, "optimal", repair_first=False)


def test_one_sided_levels_are_incomparable():
    p = lp()
    # Exactly feasible primal, infeasible dual: an upper bound only.
    x, z = [Q(1), Q(0)], [Q(-1), Q(0), Q(0)]
    assert assert_level(p, x, z, "primal_feasible", repair_first=False).level == "primal_feasible"
    with pytest.raises(AssertionError):
        assert_level(p, x, z, "dual_feasible", repair_first=False)


def test_fixture_uses_command_line_eps(certkkt):
    assert certkkt.eps == "1e-6"
    certkkt.assert_certified(lp(), [0.5, 0.5 - 1e-8], [1, 0, 0])
    with pytest.raises(AssertionError):
        certkkt.assert_certified(lp(), [0.5, 0.5 - 1e-5], [1, 0, 0])
