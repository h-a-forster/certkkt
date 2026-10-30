"""Assertion helpers for test suites of solvers and modelling code.

Example::

    from certkkt import make_problem
    from certkkt.testing import assert_certified

    def test_my_solver():
        prob = make_problem(A=..., b=..., cones=[("zero", 1), ("nonneg", 2)], q=...)
        x, z = my_solver(prob)
        assert_certified(prob, x, z, eps="1e-7")

Every check is exact: the residuals are computed in rational arithmetic on the problem
data, so a passing assertion cannot be an artefact of floating-point evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from certkkt.certify import LEVELS, Certificate, certify, first_meeting
from certkkt.problem import ConicProblem
from certkkt.repair import repair


def _describe(cert: Certificate) -> str:
    d = cert.to_dict()
    keys = (
        "level",
        "method",
        "rel_primal",
        "rel_dual",
        "rel_gap",
        "primal_eq_residual",
        "primal_cone_violation",
        "stationarity_residual",
        "dual_cone_violation",
        "pobj",
        "dobj",
        "psd",
    )
    return ", ".join(f"{k}={d[k]}" for k in keys)


def assert_certified(
    problem: ConicProblem,
    x: Sequence[Any],
    z: Sequence[Any] | None,
    *,
    eps: str | float = "1e-6",
    repair_first: bool = False,
) -> Certificate:
    """Assert that every exact relative residual is at most ``eps``.

    With ``repair_first`` the certificate is the best one found by
    :func:`certkkt.repair.repair` (useful when the claim under test is "this solution is
    within eps of an exactly optimal one" rather than "these floats satisfy eps").
    """
    if z is None:
        raise AssertionError("no dual solution: optimality cannot be certified")
    if repair_first:
        import numpy as np

        cert = repair(problem, np.asarray(x, float), np.asarray(z, float)).best
    else:
        cert = certify(problem, x, z)
    if not cert.certified_at(eps):
        raise AssertionError(f"not certified at eps={eps}: {_describe(cert)}")
    return cert


def assert_level(
    problem: ConicProblem,
    x: Sequence[Any],
    z: Sequence[Any] | None,
    level: str = "bounded",
    *,
    repair_first: bool = True,
) -> Certificate:
    """Assert an exact certificate at least as strong as ``level`` (see LEVELS)."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; choose from {LEVELS}")
    if repair_first:
        import numpy as np

        zf = None if z is None else np.asarray(z, float)
        certs = repair(problem, np.asarray(x, float), zf).certificates
    else:
        certs = [certify(problem, x, z)]
    cert = first_meeting(certs, level)
    if cert is not None:
        return cert
    raise AssertionError(f"certificate does not meet {level!r}: {_describe(certs[0])}")


def assert_primal_feasible(
    problem: ConicProblem, x: Sequence[Any], *, eps: str | float = "1e-6"
) -> Certificate:
    """Assert that the exact relative primal residual is at most ``eps``."""
    from certkkt.rational import to_q

    cert = certify(problem, x, None)
    if cert.rel_primal > to_q(eps):
        raise AssertionError(f"primal residual above eps={eps}: {_describe(cert)}")
    return cert


__all__ = ["assert_certified", "assert_level", "assert_primal_feasible"]
