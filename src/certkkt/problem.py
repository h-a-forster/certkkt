"""Problem model: the canonical conic form and the boxed (MPS/QPS) form.

Canonical conic form (the one every certificate is stated in)::

    minimize    1/2 x'Px + q'x + r
    subject to  A x + s = b,   s in K = K_1 x ... x K_p

with cones ``zero`` (s = 0), ``nonneg`` (s >= 0) and ``soc`` (s_0 >= ||s_1:||_2). Its dual
is ``maximize -1/2 x'Px - b'z + r`` subject to ``P x + q + A'z = 0`` and ``z in K*``,
where the dual of ``zero`` is the free cone and the other cones are self-dual. This is the
convention of Clarabel and SCS.

Boxed form (MPS/QPS)::

    minimize 1/2 x'Px + c'x + c0   subject to  rl <= A x <= ru,  l <= x <= u

:meth:`BoxedQP.to_conic` converts it, recording where each conic row came from so that
boxed duals can be mapped to canonical conic duals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from certkkt.rational import ONE, ZERO, Q, to_q
from certkkt.sparse import QMatrix

CONE_KINDS = ("zero", "nonneg", "soc")


@dataclass(frozen=True)
class Cone:
    """One cone of the product: ``kind`` in CONE_KINDS and its dimension."""

    kind: str
    dim: int

    def __post_init__(self) -> None:
        if self.kind not in CONE_KINDS:
            raise ValueError(f"unknown cone kind {self.kind!r}; expected one of {CONE_KINDS}")
        if self.dim < 1:
            raise ValueError(f"cone dimension must be positive, got {self.dim}")


def merge_cones(cones: Sequence[Cone]) -> list[Cone]:
    """Merge adjacent zero / nonneg cones (SOC blocks are kept separate)."""
    out: list[Cone] = []
    for c in cones:
        if out and c.kind == out[-1].kind and c.kind != "soc":
            out[-1] = Cone(c.kind, out[-1].dim + c.dim)
        else:
            out.append(c)
    return out


@dataclass
class ConicProblem:
    """Exact canonical conic problem. All data are exact rationals.

    ``P`` is stored in full (both triangles) and must be symmetric. ``objective_sign`` is
    -1 when the source problem was a maximisation that was negated into this form; the
    certificate objectives refer to this (minimisation) form.
    """

    P: QMatrix
    q: list[Any]
    A: QMatrix
    b: list[Any]
    cones: list[Cone]
    r: Any = ZERO
    name: str = "problem"
    objective_sign: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def is_lp(self) -> bool:
        return self.P.nnz == 0 and self.is_polyhedral

    @property
    def is_polyhedral(self) -> bool:
        return all(c.kind != "soc" for c in self.cones)

    @property
    def has_quadratic_objective(self) -> bool:
        return self.P.nnz > 0

    def cone_slices(self) -> list[tuple[Cone, int, int]]:
        """``(cone, start, stop)`` row ranges in order."""
        out = []
        start = 0
        for c in self.cones:
            out.append((c, start, start + c.dim))
            start += c.dim
        return out

    def validate(self) -> None:
        m, n = self.A.shape
        if self.P.shape != (n, n):
            raise ValueError(f"P has shape {self.P.shape}, expected ({n}, {n})")
        if len(self.q) != n:
            raise ValueError(f"q has length {len(self.q)}, expected {n}")
        if len(self.b) != m:
            raise ValueError(f"b has length {len(self.b)}, expected {m}")
        if sum(c.dim for c in self.cones) != m:
            raise ValueError(
                f"cone dimensions sum to {sum(c.dim for c in self.cones)}, A has {m} rows"
            )
        if self.objective_sign not in (1, -1):
            raise ValueError("objective_sign must be 1 or -1")
        if not self.P.is_symmetric():
            raise ValueError("P must be symmetric (store both triangles)")

    # -- float views for solvers ---------------------------------------------------------
    def float_data(self) -> FloatData:
        """Correctly rounded float copies of the data (what solvers receive)."""
        return FloatData(
            P=self.P.to_scipy(),
            q=_to_float(self.q),
            A=self.A.to_scipy(),
            b=_to_float(self.b),
            r=_to_float([self.r])[0],
        )

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for c in self.cones:
            kinds[c.kind] = kinds.get(c.kind, 0) + (c.dim if c.kind != "soc" else 1)
        return {
            "name": self.name,
            "n": self.n,
            "m": self.m,
            "nnz_A": self.A.nnz,
            "nnz_P": self.P.nnz,
            "cones": kinds,
            "class": problem_class(self),
        }


def problem_class(p: ConicProblem) -> str:
    if not p.is_polyhedral:
        return "QSOCP" if p.has_quadratic_objective else "SOCP"
    return "QP" if p.has_quadratic_objective else "LP"


@dataclass
class FloatData:
    P: sp.csr_array
    q: np.ndarray
    A: sp.csr_array
    b: np.ndarray
    r: float


def _to_float(values: Sequence[Any]) -> np.ndarray:
    return np.array([int(v.numerator) / int(v.denominator) for v in values], dtype=np.float64)


# -- boxed form --------------------------------------------------------------------------


@dataclass(frozen=True)
class RowOrigin:
    """Where a conic row came from: ``kind`` "row" or "col", index, and side.

    side: "eq" (equality), "lo" (lower bound) or "up" (upper bound). The conic row for a
    lower bound reads ``-a x + s = -lo``; for an upper bound ``a x + s = up``; for an
    equality ``a x + s = rhs`` with ``s = 0``.
    """

    kind: str
    index: int
    side: str


@dataclass
class BoxedQP:
    """Exact boxed QP. ``None`` in a bound vector means infinite."""

    P: QMatrix
    c: list[Any]
    A: QMatrix
    rl: list[Any | None]
    ru: list[Any | None]
    lb: list[Any | None]
    ub: list[Any | None]
    c0: Any = ZERO
    name: str = "problem"
    objective_sign: int = 1
    row_names: list[str] = field(default_factory=list)
    col_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        m, n = self.A.shape
        if self.P.shape != (n, n):
            raise ValueError("P shape mismatch")
        if not (len(self.c) == len(self.lb) == len(self.ub) == n):
            raise ValueError("column vector length mismatch")
        if not (len(self.rl) == len(self.ru) == m):
            raise ValueError("row vector length mismatch")
        for kind, lo, hi in (("row", self.rl, self.ru), ("column", self.lb, self.ub)):
            for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
                if a is not None and b is not None and a > b:
                    raise ValueError(f"{kind} {i} has lower bound {a} > upper bound {b}")
        if not self.P.is_symmetric():
            raise ValueError("P must be symmetric")

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    def to_conic(self) -> tuple[ConicProblem, list[RowOrigin]]:
        """Conic form with zero-cone rows first, then nonneg rows, plus row origins."""
        eq_rows: list[dict[int, Any]] = []
        eq_b: list[Any] = []
        eq_org: list[RowOrigin] = []
        in_rows: list[dict[int, Any]] = []
        in_b: list[Any] = []
        in_org: list[RowOrigin] = []
        arows = self.A.row_dicts()
        for i, row in enumerate(arows):
            lo, hi = self.rl[i], self.ru[i]
            if lo is not None and hi is not None and lo == hi:
                eq_rows.append(dict(row))
                eq_b.append(lo)
                eq_org.append(RowOrigin("row", i, "eq"))
                continue
            if lo is not None:
                in_rows.append({j: -v for j, v in row.items()})
                in_b.append(-lo)
                in_org.append(RowOrigin("row", i, "lo"))
            if hi is not None:
                in_rows.append(dict(row))
                in_b.append(hi)
                in_org.append(RowOrigin("row", i, "up"))
        for j in range(self.n):
            lo, hi = self.lb[j], self.ub[j]
            if lo is not None and hi is not None and lo == hi:
                eq_rows.append({j: ONE})
                eq_b.append(lo)
                eq_org.append(RowOrigin("col", j, "eq"))
                continue
            if lo is not None:
                in_rows.append({j: -ONE})
                in_b.append(-lo)
                in_org.append(RowOrigin("col", j, "lo"))
            if hi is not None:
                in_rows.append({j: ONE})
                in_b.append(hi)
                in_org.append(RowOrigin("col", j, "up"))
        rows = eq_rows + in_rows
        cones = []
        if eq_rows:
            cones.append(Cone("zero", len(eq_rows)))
        if in_rows:
            cones.append(Cone("nonneg", len(in_rows)))
        A = QMatrix.from_rows((len(rows), self.n), rows)
        prob = ConicProblem(
            P=self.P,
            q=list(self.c),
            A=A,
            b=eq_b + in_b,
            cones=cones,
            r=self.c0,
            name=self.name,
            objective_sign=self.objective_sign,
            meta={"warnings": list(self.warnings)} if self.warnings else {},
        )
        return prob, eq_org + in_org

    def float_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(rl, ru, lb, ub) as floats with +-inf for missing bounds."""

        def conv(vals: list[Any | None], fill: float) -> np.ndarray:
            return np.array(
                [fill if v is None else int(v.numerator) / int(v.denominator) for v in vals],
                dtype=np.float64,
            )

        return (
            conv(self.rl, -np.inf),
            conv(self.ru, np.inf),
            conv(self.lb, -np.inf),
            conv(self.ub, np.inf),
        )


def boxed_duals_to_conic(
    origins: Sequence[RowOrigin], y_row: Sequence[float], y_col: Sequence[float]
) -> np.ndarray:
    """Map boxed multipliers to canonical conic duals ``z`` (float).

    Boxed convention (OSQP's): stationarity ``P x + c + A'y_row + y_col = 0`` with
    ``y > 0`` where an upper bound is active and ``y < 0`` where a lower bound is active.
    A two-sided row splits ``y`` by sign; a one-sided row passes ``y`` through unclipped
    (a wrong-signed multiplier shows up as a dual cone violation, not silently dropped).
    """
    y_row = np.asarray(y_row, dtype=np.float64)
    y_col = np.asarray(y_col, dtype=np.float64)
    sides: dict[tuple[str, int], set[str]] = {}
    for o in origins:
        sides.setdefault((o.kind, o.index), set()).add(o.side)
    z = np.zeros(len(origins))
    for k, o in enumerate(origins):
        y = float(y_row[o.index] if o.kind == "row" else y_col[o.index])
        two_sided = sides[(o.kind, o.index)] == {"lo", "up"}
        if o.side == "eq" or o.side == "up":
            z[k] = max(y, 0.0) if two_sided else y
        else:  # lower bound row: -a x + s = -lo, contributes -a z
            z[k] = max(-y, 0.0) if two_sided else -y
    return z


def conic_duals_to_boxed(
    origins: Sequence[RowOrigin], z: Sequence[Any], m: int, n: int
) -> tuple[list[Any], list[Any]]:
    """Inverse of :func:`boxed_duals_to_conic` for exact duals (sums the two sides)."""
    y_row = [ZERO] * m
    y_col = [ZERO] * n
    for k, o in enumerate(origins):
        zk = to_q(z[k])
        target = y_row if o.kind == "row" else y_col
        target[o.index] += -zk if o.side == "lo" else zk
    return y_row, y_col


def make_problem(
    *,
    A: Any,
    b: Sequence[object],
    cones: Sequence[Cone | tuple[str, int]],
    q: Sequence[object] | None = None,
    P: Any = None,
    r: object = 0,
    name: str = "problem",
) -> ConicProblem:
    """Convenience constructor from dense/sparse floats, ints, strings or rationals.

    ``P`` may be given as its upper triangle only (as Clarabel and OSQP take it) or in
    full; an upper-triangular input is symmetrised.
    """
    Aq = A if isinstance(A, QMatrix) else _qmatrix(A)
    n = Aq.shape[1]
    if P is None:
        Pq = QMatrix.zeros((n, n))
    else:
        Pq = P if isinstance(P, QMatrix) else _qmatrix(P)
        if not Pq.is_symmetric():
            Pq = _symmetrize_upper(Pq)
    qq = [to_q(v) for v in (q if q is not None else [0] * n)]
    cone_list = [c if isinstance(c, Cone) else Cone(c[0], int(c[1])) for c in cones]
    return ConicProblem(
        P=Pq, q=qq, A=Aq, b=[to_q(v) for v in b], cones=cone_list, r=to_q(r), name=name
    )


def _qmatrix(mat: Any) -> QMatrix:
    if sp.issparse(mat):
        return QMatrix.from_scipy(mat)
    arr = mat if isinstance(mat, list) else np.asarray(mat).tolist()
    if arr and not isinstance(arr[0], list):
        arr = [arr]
    rows = []
    for row in arr:
        rows.append({j: to_q(v) for j, v in enumerate(row) if to_q(v)})
    ncols = len(arr[0]) if arr else 0
    return QMatrix.from_rows((len(arr), ncols), rows)


def _symmetrize_upper(P: QMatrix) -> QMatrix:
    rows = P.row_dicts()
    for i, row in enumerate(rows):
        for j in row:
            if j < i:
                raise ValueError("P is neither symmetric nor upper triangular")
    full: list[dict[int, Any]] = [dict(r) for r in rows]
    for i, row in enumerate(rows):
        for j, v in row.items():
            if j != i:
                full[j][i] = v
    return QMatrix.from_rows(P.shape, full)


def upper_triangle(P: sp.csr_array | sp.spmatrix) -> sp.csc_array:
    """Upper triangle of a symmetric float matrix in CSC form (solver input format)."""
    return sp.csc_array(sp.triu(P, format="csc"))


__all__ = [
    "BoxedQP",
    "CONE_KINDS",
    "Cone",
    "ConicProblem",
    "FloatData",
    "Q",
    "RowOrigin",
    "boxed_duals_to_conic",
    "conic_duals_to_boxed",
    "make_problem",
    "merge_cones",
    "problem_class",
    "upper_triangle",
]
