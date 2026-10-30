"""Minimum-variance portfolio SOCPs built from Ken French industry portfolio returns.

For ``K`` industries, a window of ``T`` monthly returns ending at ``end`` (YYYYMM), a gross
leverage cap ``L`` and an L2 turnover budget ``tau`` around the equal-weight portfolio
``w0 = 1/K``::

    minimize    t
    subject to  || R w ||_2 <= t            (risk cone; R = centred returns, T x K)
                sum(w) = 1
                -u <= w <= u,  sum(u) <= L   (gross leverage ||w||_1 <= L)
                || w - w0 ||_2 <= tau        (turnover cone)

``t`` is ``sqrt(T - 1)`` times the in-sample volatility of ``w`` (in percent). Returns are
the published two-decimal percentages, read as exact decimals, so the problem data are
exact rationals (the centring divides by ``T``). Every instance is feasible: ``w0`` itself
satisfies all constraints when ``L >= 1``.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from certkkt.instance import Instance
from certkkt.problem import Cone, ConicProblem
from certkkt.rational import ONE, ZERO, Q, to_q
from certkkt.sparse import QMatrix

MISSING = (to_q("-99.99"), to_q("-999"))


@dataclass(frozen=True)
class Returns:
    """Monthly value-weighted returns (percent, exact) by YYYYMM."""

    industries: list[str]
    months: list[int]
    values: list[list[Any]]

    def window(self, end: int, length: int) -> list[list[Any]]:
        if end not in self.months:
            raise ValueError(f"month {end} not in the data")
        k = self.months.index(end) + 1
        if k < length:
            raise ValueError(f"only {k} months up to {end}, need {length}")
        rows = self.values[k - length : k]
        if any(v in MISSING for row in rows for v in row):
            raise ValueError(f"missing returns in the window ending {end}")
        return rows


def read_french_zip(path: str | Path) -> Returns:
    """Parse the first table (value-weighted monthly returns) of a French CSV zip."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one CSV member, found {names}")
        text = zf.read(names[0]).decode("latin-1")
    return parse_french_csv(text)


def parse_french_csv(text: str) -> Returns:
    lines = io.StringIO(text).read().splitlines()
    start = None
    for i, line in enumerate(lines):
        if "Value Weighted Returns" in line and "Monthly" in line:
            start = i
            break
    if start is None:
        raise ValueError("no 'Value Weighted Returns -- Monthly' table")
    header = [h.strip() for h in lines[start + 1].split(",")][1:]
    months: list[int] = []
    values: list[list[Any]] = []
    for line in lines[start + 2 :]:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 6:
            break
        if len(parts) != len(header) + 1:
            raise ValueError(f"row {parts[0]} has {len(parts) - 1} values, expected {len(header)}")
        months.append(int(parts[0]))
        values.append([to_q(p) for p in parts[1:]])
    if not months:
        raise ValueError("empty returns table")
    return Returns(header, months, values)


def min_variance_socp(
    returns: list[list[Any]], leverage: object, turnover: object, name: str = "portfolio"
) -> ConicProblem:
    """The SOCP of the module docstring for a ``T x K`` window of exact returns."""
    T, K = len(returns), len(returns[0])
    lev, tau = to_q(leverage), to_q(turnover)
    if lev < 1:
        raise ValueError("leverage must be at least 1 (otherwise sum(w) = 1 is infeasible)")
    if tau <= 0:
        raise ValueError("turnover budget must be positive")
    means = [sum((row[i] for row in returns), ZERO) / T for i in range(K)]
    n = 2 * K + 1  # w (K), u (K), t
    it = 2 * K
    rows: list[dict[int, Any]] = []
    b: list[Any] = []
    cones: list[Cone] = []
    # zero: sum(w) = 1
    rows.append({i: ONE for i in range(K)})
    b.append(ONE)
    cones.append(Cone("zero", 1))
    # nonneg: u - w >= 0, u + w >= 0, L - sum(u) >= 0   (row of A is -(coefficients))
    for i in range(K):
        rows.append({i: ONE, K + i: -ONE})
    for i in range(K):
        rows.append({i: -ONE, K + i: -ONE})
    rows.append({K + i: ONE for i in range(K)})
    b.extend([ZERO] * (2 * K) + [lev])
    cones.append(Cone("nonneg", 2 * K + 1))
    # soc: (t, R w)
    rows.append({it: -ONE})
    b.append(ZERO)
    for row in returns:
        rows.append({i: -(row[i] - means[i]) for i in range(K) if row[i] != means[i]})
        b.append(ZERO)
    cones.append(Cone("soc", T + 1))
    # soc: (tau, w - w0)
    rows.append({})
    b.append(tau)
    w0 = Q(1, K)
    for i in range(K):
        rows.append({i: -ONE})
        b.append(-w0)
    cones.append(Cone("soc", K + 1))
    q = [ZERO] * n
    q[it] = ONE
    return ConicProblem(
        P=QMatrix.zeros((n, n)),
        q=q,
        A=QMatrix.from_rows((len(rows), n), rows),
        b=b,
        cones=cones,
        name=name,
    )


@dataclass(frozen=True)
class PortfolioSpec:
    industries: int
    end: int
    leverage: str
    turnover: str
    window: int = 60

    @property
    def name(self) -> str:
        return f"pf{self.industries}_{self.end}_L{self.leverage}_T{self.turnover}"


#: The family studied (DESIGN section 5.3).
FAMILY = tuple(
    PortfolioSpec(k, end, lev, "0.25")
    for k in (10, 30, 49)
    for end in range(198512, 202513, 500)
    for lev in ("1.3", "2")
)


def build_instance(spec: PortfolioSpec, data_dir: Path) -> Instance:
    from certkkt.datasets import FRENCH_FILES

    rets = read_french_zip(data_dir / FRENCH_FILES[spec.industries])
    window = rets.window(spec.end, spec.window)
    prob = min_variance_socp(window, spec.leverage, spec.turnover, name=spec.name)
    return Instance(problem=prob, source=FRENCH_FILES[spec.industries])


__all__ = [
    "FAMILY",
    "PortfolioSpec",
    "Returns",
    "build_instance",
    "min_variance_socp",
    "parse_french_csv",
    "read_french_zip",
]
