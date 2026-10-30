"""Turn a near-optimal float solution into an exact certificate.

Every repair produces a rational candidate ``(x, z)`` that is then certified exactly by
:func:`certkkt.certify.certify`; a repair can make a certificate stronger but never makes
an invalid claim, because nothing is trusted until it is re-checked in exact arithmetic.
:func:`repair` runs the methods in order and returns every certificate, best first.

Methods (DESIGN section 4):

``as_is``
    The solver's floats, converted exactly.
``fixed``
    Exact equality restoration of the primal (solve ``A_Z d = b_Z - A_Z x`` exactly on
    columns with slack) and an exact dual fix: project z onto K*, then absorb the
    stationarity residual into multipliers of singleton (bound) rows, as in the safe
    dual bounds of Neumaier and Shcherbina, and finally solve for any remaining residual
    exactly on rows whose multipliers have room.
``snapped``
    Each coordinate replaced by the simplest rational within a relative tolerance, then
    ``fixed``. Recovers exact optima with small denominators.
``active_set`` (polyhedral problems)
    Guess the active set from the float pair and solve the KKT system restricted to it
    exactly (exact LP/QP solving by basis verification). Success gives gap exactly 0.
``interior``
    Push the primal slack and the dual into the interior of their cones by a relative
    margin ``eta``, project back onto the affine constraints in floating point (the dual
    step weighted by the multipliers, a Dikin-ellipsoid-style step as in the
    reconstruction formula of Lee, Papp and Varga), then restore the affine constraints
    exactly. Needed for SOCPs, whose optima are generally irrational.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from certkkt.certify import Certificate, certify, psd_status
from certkkt.problem import ConicProblem
from certkkt.rational import ZERO, Q, snap, sqrt_up, to_q
from certkkt.sparse import DEFAULT_BUDGET, BudgetExceeded, solve_sparse

METHODS = ("as_is", "fixed", "snapped", "active_set", "interior")
SNAP_TOLS = (1e-12, 1e-9)
INTERIOR_ETAS = (1e-9, 1e-7, 1e-5)


@dataclass(frozen=True)
class Limits:
    """Work limits for exact solves: scalar-operation budget and a perf_counter deadline."""

    ops: int = DEFAULT_BUDGET
    deadline: float | None = None


@dataclass
class RepairOutcome:
    """All certificates (best first) plus per-method notes and timings."""

    certificates: list[Certificate]
    notes: dict[str, str] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> Certificate:
        return self.certificates[0]

    def by_method(self, method: str) -> Certificate | None:
        for c in self.certificates:
            if c.method == method:
                return c
        return None


def repair(
    problem: ConicProblem,
    x: np.ndarray,
    z: np.ndarray | None,
    *,
    methods: Sequence[str] = METHODS,
    budget: int = DEFAULT_BUDGET,
    time_budget: float = 120.0,
    stop_at_optimal: bool = True,
) -> RepairOutcome:
    """Certify the float pair as is and after each repair method; best certificate first.

    ``time_budget`` (seconds) bounds the exact solves of all methods together; a method
    that runs out of time or operations (``budget`` per elimination) is noted and its
    candidates dropped. Certification itself is linear in the data size and not bounded.
    """
    for mth in methods:
        if mth not in METHODS:
            raise ValueError(f"unknown repair method {mth!r}; choose from {METHODS}")
    x = np.asarray(x, dtype=np.float64)
    zf = None if z is None else np.asarray(z, dtype=np.float64)
    t_start = time.perf_counter()
    lim = Limits(budget, t_start + time_budget)
    certs: list[Certificate] = []
    notes: dict[str, str] = {}
    seconds: dict[str, float] = {}
    psd_status(problem, budget)  # computed once, reused by every certificate

    def done() -> bool:
        return stop_at_optimal and any(c.level == "optimal" for c in certs)

    for mth in methods:
        if done():
            notes.setdefault(mth, "skipped: an exactly optimal certificate was already found")
            continue
        if time.perf_counter() - t_start > time_budget:
            notes[mth] = "skipped: time budget exhausted"
            continue
        t0 = time.perf_counter()
        try:
            new = _run(mth, problem, x, zf, lim)
        except BudgetExceeded as exc:
            notes[mth] = f"budget: {exc}"
            new = []
        seconds[mth] = time.perf_counter() - t0
        if isinstance(new, str):
            notes[mth] = new
            continue
        certs.extend(new)
    if not certs:
        raise RuntimeError("no certificate could be computed")
    certs.sort(key=lambda c: c.rank())
    return RepairOutcome(certs, notes, seconds)


def _run(
    mth: str, problem: ConicProblem, x: np.ndarray, z: np.ndarray | None, lim: Limits
) -> list[Certificate] | str:
    if mth == "as_is":
        return [certify(problem, x, z, method="as_is")]
    xq = [to_q(v) for v in x.tolist()]
    zq = None if z is None else [to_q(v) for v in z.tolist()]
    if mth == "fixed":
        xr, zr = fix_pair(problem, xq, zq, x, lim)
        return [certify(problem, xr, zr, method="fixed")]
    if mth == "snapped":
        out = []
        for tol in SNAP_TOLS:
            xs = [snap(v, tol) for v in x.tolist()]
            zs = None if z is None else [snap(v, tol) for v in z.tolist()]
            xr, zr = fix_pair(problem, xs, zs, x, lim)
            out.append(certify(problem, xr, zr, method=f"snapped@{tol:g}"))
        return out
    if mth == "active_set":
        if not problem.is_polyhedral:
            return "not applicable: problem has second-order cones"
        if z is None:
            return "not applicable: no dual solution"
        out = []
        for rule in ("complementarity", "threshold"):
            pair = active_set_solve(problem, x, z, rule, lim)
            if pair is None:
                continue
            out.append(certify(problem, pair[0], pair[1], method=f"active_set/{rule}"))
        return out or "active-set KKT systems were inconsistent"
    if mth == "interior":
        out = []
        for eta in INTERIOR_ETAS:
            xi = interior_primal(problem, x, eta)
            xr = restore_equalities(problem, [to_q(v) for v in xi.tolist()], xi, lim)
            if xr is None:
                xr = [to_q(v) for v in xi.tolist()]
            zr = None
            if z is not None:
                zi = interior_dual(problem, xr, z, eta)
                zr = fix_dual(problem, xr, [to_q(v) for v in zi.tolist()], lim)
            out.append(certify(problem, xr, zr, method=f"interior@{eta:g}"))
        return out
    raise ValueError(mth)


# -- primal --------------------------------------------------------------------------------


def _row_lists(problem: ConicProblem) -> tuple[list[int], list[int]]:
    zero, other = [], []
    for cone, a, b in problem.cone_slices():
        (zero if cone.kind == "zero" else other).extend(range(a, b))
    return zero, other


def restore_equalities(
    problem: ConicProblem, xq: list[Any], xf: np.ndarray, lim: Limits
) -> list[Any] | None:
    """Exactly satisfy the zero-cone rows by an exact correction on columns with slack.

    Columns whose inequality rows are all far from active get priority as pivots, so the
    (tiny) correction does not push active inequalities out of their cones. Returns None
    if the equality system is inconsistent.
    """
    zero_rows, other_rows = _row_lists(problem)
    if not zero_rows:
        return list(xq)
    A = problem.A
    resid = []
    any_nonzero = False
    for i in zero_rows:
        cols, vals = A.row(i)
        r = problem.b[i]
        for j, v in zip(cols, vals, strict=True):
            if xq[j]:
                r -= v * xq[j]
        resid.append(r)
        any_nonzero = any_nonzero or bool(r)
    if not any_nonzero:
        return list(xq)
    # Penalise columns that appear in nearly active inequality rows (float estimate).
    fd = problem.float_data()
    s = fd.b - fd.A @ xf
    scale = 1.0 + np.abs(fd.b)
    tight = np.zeros(problem.n, dtype=bool)
    near = [i for i in other_rows if s[i] <= 1e-6 * scale[i]]
    if near:
        tight[np.unique(fd.A[near].indices)] = True
    penalty = {j: 1 for j in np.flatnonzero(tight).tolist()}
    rows = [dict(zip(*A.row(i), strict=True)) for i in zero_rows]
    d = solve_sparse(
        rows, resid, problem.n, col_penalty=penalty, budget=lim.ops, deadline=lim.deadline
    )
    if d is None:
        return None
    return [a + b if b else a for a, b in zip(xq, d, strict=True)]


def interior_primal(problem: ConicProblem, x: np.ndarray, eta: float) -> np.ndarray:
    """Float x whose slack is pushed into the interior of K by a relative margin eta.

    Rows that need a push get target slacks; ``min ||d|| s.t. A_R d = s_R - t_R`` (with
    zero rows held fixed) is solved by LSQR.
    """
    fd = problem.float_data()
    s = fd.b - fd.A @ x
    margin = eta * (1.0 + np.abs(fd.b))  # per-row relative margin
    target = s.copy()
    rows: list[int] = []
    for cone, a, b in problem.cone_slices():
        if cone.kind == "zero":
            target[a:b] = 0.0
            rows.extend(range(a, b))
        elif cone.kind == "nonneg":
            idx = np.flatnonzero(s[a:b] < margin[a:b]) + a
            target[idx] = margin[idx]
            rows.extend(idx.tolist())
        else:
            t, v = s[a], s[a + 1 : b]
            need = float(np.linalg.norm(v)) + float(margin[a:b].max())
            if t < need:
                target[a] = need
                rows.extend(range(a, b))
    if not rows:
        return x.copy()
    rows_arr = np.array(sorted(set(rows)), dtype=np.int64)
    AR = sp.csr_matrix(fd.A[rows_arr])
    rhs = s[rows_arr] - target[rows_arr]
    d = spla.lsqr(AR, rhs, atol=1e-15, btol=1e-15, iter_lim=20 * AR.shape[1] + 100)[0]
    return x + d


# -- dual ----------------------------------------------------------------------------------


def project_dual_cone(problem: ConicProblem, zq: list[Any]) -> list[Any]:
    """Exact rational point of K* near z: clip nonneg rows, lift SOC heads to cover tails."""
    out = list(zq)
    for cone, a, b in problem.cone_slices():
        if cone.kind == "nonneg":
            for i in range(a, b):
                if out[i] < 0:
                    out[i] = ZERO
        elif cone.kind == "soc":
            sq = sum((out[i] * out[i] for i in range(a + 1, b) if out[i]), ZERO)
            if out[a] < 0 or out[a] * out[a] < sq:
                out[a] = sqrt_up(sq)
    return out


def fix_dual(
    problem: ConicProblem,
    xq: list[Any],
    zq: list[Any],
    lim: Limits,
) -> list[Any]:
    """Exact dual repair for a fixed x (see module docstring). Always returns a vector;
    whether it is exactly dual feasible is for :func:`certify` to decide."""
    z = project_dual_cone(problem, zq)
    A = problem.A
    px = problem.P.matvec(xq)
    atz = A.rmatvec(z)
    rho = [p + q + a for p, q, a in zip(px, problem.q, atz, strict=True)]
    if not any(rho):
        return z
    kinds = _row_kinds(problem)
    # Singleton rows per column.
    single: dict[int, list[tuple[int, Any]]] = {}
    for k in range(problem.m):
        if kinds[k] == "soc":
            continue
        a, b = A.indptr[k], A.indptr[k + 1]
        if b - a == 1:
            single.setdefault(A.indices[a], []).append((k, A.data[a]))
    for j, r in enumerate(rho):
        if not r or j not in single:
            continue
        # Need sum_k a_kj dz_k = -r. Free rows first, then shrink an existing multiplier,
        # then grow one whose sign helps.
        for k, a in sorted(single[j], key=lambda t: (kinds[t[0]] != "zero", -abs(z[t[0]]))):
            if not r:
                break
            dz = -r / a
            if kinds[k] == "zero" or z[k] + dz >= 0:
                z[k] += dz
                r = ZERO
            elif z[k] > 0:  # shrink to zero, keep the remainder
                r -= a * z[k]
                z[k] = ZERO
        rho[j] = r
    if not any(rho):
        return z
    # General exact solve on rows whose multipliers have room.
    penalty: dict[int, int] = {}
    scale = max((abs(v) for v in z), default=ZERO)
    tiny = scale * Q(1, 10**6) if scale else Q(1, 10**12)
    for k in range(problem.m):
        if kinds[k] == "zero":
            continue
        penalty[k] = 0 if kinds[k] == "nonneg" and z[k] > tiny else (1 if kinds[k] == "soc" else 2)
    cols: list[dict[int, Any]] = [{} for _ in range(problem.n)]
    for k in range(problem.m):
        for p in range(A.indptr[k], A.indptr[k + 1]):
            cols[A.indices[p]][k] = A.data[p]
    d = solve_sparse(
        cols,
        [-r for r in rho],
        problem.m,
        col_penalty=penalty,
        budget=lim.ops,
        deadline=lim.deadline,
    )
    if d is None:
        return z
    return [a + b if b else a for a, b in zip(z, d, strict=True)]


def _row_kinds(problem: ConicProblem) -> list[str]:
    kinds = []
    for cone in problem.cones:
        kinds.extend([cone.kind] * cone.dim)
    return kinds


def interior_dual(problem: ConicProblem, xq: list[Any], z: np.ndarray, eta: float) -> np.ndarray:
    """Float z pushed into int K* by a margin, then corrected toward ``A'z = -(Px + q)``
    with a step weighted by the multipliers (large multipliers move most).

    Row i gets margin ``eta (1 + ||q||) / (m (1 + |s_i|))``, so the pushes add at most
    ``eta (1 + ||q||)`` to the complementarity ``s'z``.
    """
    fd = problem.float_data()
    xf = np.array([int(v.numerator) / int(v.denominator) for v in xq])
    g = -(fd.P @ xf + fd.q)
    s = np.abs(fd.b - fd.A @ xf)
    margin = eta * (1.0 + np.abs(fd.q).max(initial=0.0)) / (max(problem.m, 1) * (1.0 + s))
    u = z.copy()
    w = np.ones_like(u)
    for cone, a, b in problem.cone_slices():
        if cone.kind == "nonneg":
            u[a:b] = np.maximum(u[a:b], margin[a:b])
            w[a:b] = u[a:b]
        elif cone.kind == "soc":
            need = float(np.linalg.norm(u[a + 1 : b])) + float(margin[a:b].min())
            if u[a] < need:
                u[a] = need
            w[a:b] = u[a] - float(np.linalg.norm(u[a + 1 : b]))  # distance to the boundary
        else:
            w[a:b] = max(float(np.abs(u).max(initial=1.0)), 1.0)
    rho = g - fd.A.T @ u
    if not np.any(rho):
        return u
    AtW = sp.csr_matrix(fd.A.T @ sp.diags(w))
    dd = spla.lsqr(AtW, rho, atol=1e-15, btol=1e-15, iter_lim=20 * AtW.shape[1] + 100)[0]
    return u + w * dd


def fix_pair(
    problem: ConicProblem,
    xq: list[Any],
    zq: list[Any] | None,
    xf: np.ndarray,
    lim: Limits,
) -> tuple[list[Any], list[Any] | None]:
    """Equality restoration of x, then the exact dual fix for that x."""
    xr = restore_equalities(problem, xq, xf, lim)
    if xr is None:
        xr = xq
    zr = None if zq is None else fix_dual(problem, xr, zq, lim)
    return xr, zr


# -- active set ----------------------------------------------------------------------------


def active_set_solve(
    problem: ConicProblem, x: np.ndarray, z: np.ndarray, rule: str, lim: Limits
) -> tuple[list[Any], list[Any]] | None:
    """Solve the KKT system for a guessed active set exactly (polyhedral problems).

    rule "complementarity": a nonneg row is active when its multiplier exceeds its slack;
    rule "threshold": when its slack is below ``1e-7 (1 + |b_i|)``. Unknowns are x and
    the multipliers of zero rows and active rows; inactive multipliers are zero.
    """
    fd = problem.float_data()
    s = fd.b - fd.A @ x
    zero_rows, other_rows = _row_lists(problem)
    if rule == "complementarity":
        act = [i for i in other_rows if z[i] > s[i]]
    elif rule == "threshold":
        act = [i for i in other_rows if s[i] <= 1e-7 * (1.0 + abs(fd.b[i]))]
    else:
        raise ValueError(rule)
    rows_used = zero_rows + act
    n = problem.n
    nz = len(rows_used)
    A = problem.A
    # Equations: P x + sum_k A_k' z_k = -q (n rows), A_k x = b_k for used rows.
    eqs: list[dict[int, Any]] = [dict() for _ in range(n + nz)]
    prow = problem.P.row_dicts()
    for j in range(n):
        eqs[j].update(prow[j])
    for t, k in enumerate(rows_used):
        cols, vals = A.row(k)
        for j, v in zip(cols, vals, strict=True):
            eqs[j][n + t] = v
            eqs[n + t][j] = v
    rhs = [-v for v in problem.q] + [problem.b[k] for k in rows_used]
    sol = solve_sparse(eqs, rhs, n + nz, budget=lim.ops, deadline=lim.deadline)
    if sol is None:
        return None
    xq = sol[:n]
    zq = [ZERO] * problem.m
    for t, k in enumerate(rows_used):
        zq[k] = sol[n + t]
    return xq, zq


__all__ = [
    "INTERIOR_ETAS",
    "METHODS",
    "SNAP_TOLS",
    "Limits",
    "RepairOutcome",
    "active_set_solve",
    "fix_dual",
    "fix_pair",
    "interior_dual",
    "interior_primal",
    "project_dual_cone",
    "repair",
    "restore_equalities",
]
