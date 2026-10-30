"""pytest plugin for certkkt (registered through the ``pytest11`` entry point).

Adds ``--certkkt-eps`` (default 1e-6) and a ``certkkt`` fixture whose methods are the
helpers of :mod:`certkkt.testing` with that tolerance as default::

    def test_solver(certkkt):
        certkkt.assert_certified(problem, x, z)

It is a top-level module, not part of the ``certkkt`` package, and imports certkkt only
when a fixture method runs: pytest loads it in every session of the environment, and
importing the package (numpy, scipy, gmpy2, the ``CERTKKT_RATIONAL`` check) would slow
down or break unrelated test suites.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from certkkt.certify import Certificate
    from certkkt.problem import ConicProblem


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("certkkt")
    group.addoption(
        "--certkkt-eps",
        default="1e-6",
        help="default relative tolerance for certkkt assertions (exact decimal)",
    )


class CertkktFixture:
    def __init__(self, eps: str) -> None:
        self.eps = eps

    def assert_certified(
        self,
        problem: ConicProblem,
        x: Sequence[Any],
        z: Sequence[Any] | None,
        *,
        eps: str | float | None = None,
        repair_first: bool = False,
    ) -> Certificate:
        from certkkt import testing

        return testing.assert_certified(
            problem, x, z, eps=self.eps if eps is None else eps, repair_first=repair_first
        )

    def assert_level(
        self, problem: ConicProblem, x: Sequence[Any], z: Sequence[Any] | None, level: str
    ) -> Certificate:
        from certkkt import testing

        return testing.assert_level(problem, x, z, level)

    def assert_primal_feasible(
        self, problem: ConicProblem, x: Sequence[Any], *, eps: str | float | None = None
    ) -> Certificate:
        from certkkt import testing

        return testing.assert_primal_feasible(problem, x, eps=self.eps if eps is None else eps)


@pytest.fixture
def certkkt(request: pytest.FixtureRequest) -> CertkktFixture:
    """Exact KKT assertions with the session's default tolerance."""
    return CertkktFixture(str(request.config.getoption("--certkkt-eps")))
