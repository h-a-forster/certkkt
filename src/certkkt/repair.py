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

import math
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
from certkkt.sparse import DEFAULT_BUDGET, BudgetExceeded, check_deadline, solve_sparse

METHODS = ("as_is", "fixed", "snapped", "active_set", "interior")
#: Default order: the method most likely to give the strongest certificate first, so it
#: gets the time budget (exact optimum by active set for LP/QP; interior for SOCP).
POLYHEDRAL_ORDER = ("as_is", "active_set", "fixed", "snapped", "interior")
CONIC_ORDER = ("as_is", "interior", "fixed", "snapped")
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
    methods: Sequence[str] | None = None,
    budget: int = DEFAULT_BUDGET,
    time_budget: float = 120.0,
    stop_at_optimal: bool = True,
) -> RepairOutcome:
    """Certify the float pair as is and after each repair method; best certificate first.

    ``time_budget`` (seconds) bounds the exact solves of all methods together; a method
    that runs out of time or operations (``budget`` per elimination) is noted and its
    candidates dropped, as are those of a method that fails numerically. Certification
    itself is linear in the data size and not bounded.
    """
    if methods is None:
        methods = POLYHEDRAL_ORDER if problem.is_polyhedral else CONIC_ORDER
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
    psd_status(problem)  # computed once, reused by every certificate

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
        except (ArithmeticError, ValueError) as exc:
            if mth == "as_is":
                raise
            # Float steps can overflow on wild input; only this method's candidates are lost.
            notes[mth] = f"failed: {type(exc).__name__}: {exc}"
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


def _room(problem: ConicProblem, v: np.ndarray) -> np.ndarray:
    """Float distance of each row of v to the boundary of its cone (inf for zero rows):
    v_i for nonneg rows, v_0 - ||v_1:|| for every row of an SOC block."""
    out = np.full(problem.m, np.inf)
    for cone, a, b in problem.cone_slices():
        if cone.kind == "nonneg":
            out[a:b] = v[a:b]
        elif cone.kind == "soc":
            out[a:b] = v[a] - float(np.linalg.norm(v[a + 1 : b]))
    return out


def weighted_min_norm(
    rows: Sequence[dict[int, Any]],
    rhs: Sequence[Any],
    weights: dict[int, float],
    lim: Limits,
    refine: int = 8,
) -> dict[int, Any] | None:
    """An exact ``d`` with ``R d = rhs`` that is close to the weighted least-norm
    correction ``D R' (R D R')^+ rhs``.

    ``rows`` are the rows of R as ``{col: value}``; ``weights`` maps a column to its weight
    D_j > 0, and columns without a weight never move. This is the reconstruction step of
    Lee, Papp and Varga (a Dikin-ellipsoid-like step when D is the distance to the cone
    boundary), computed in two stages so that no large exact system is solved: a sparse
    LU factorisation of ``R D R'`` gives the float step, refined against the exact residual
    ``refine`` times, and the remaining residual (tiny, so its correction is tiny however
    ill-conditioned the basis) is removed by an exact basic solution on the columns with
    the largest weights. Returns None if the system is inconsistent.
    """
    k = len(rows)
    cols = sorted(weights)
    pos = {j: t for t, j in enumerate(cols)}
    ri, ci, vals = [], [], []
    for a, row in enumerate(rows):
        for j, v in row.items():
            if j in pos:
                ri.append(a)
                ci.append(pos[j])
                vals.append(float(v))
    dv = np.array([weights[j] for j in cols], dtype=np.float64)
    R = sp.csr_matrix((np.array(vals), (ri, ci)), shape=(k, len(cols)))
    M = sp.csc_matrix(R @ sp.diags(dv) @ R.T)
    # A tiny diagonal shift keeps rank-deficient normal equations factorisable; the
    # exact residual is recomputed after each step, so it cannot cause a wrong answer.
    shift = 1e-13 * max(float(M.diagonal().max(initial=0.0)), 1e-300)
    try:
        lu = spla.splu(M + shift * sp.identity(k, format="csc"))
    except RuntimeError:
        lu = None
    d: dict[int, Any] = {}
    resid = list(rhs)
    prev = None
    for _ in range(refine if lu is not None else 0):
        rmax = max((abs(v) for v in resid), default=ZERO)
        if not rmax:
            return d
        if prev is not None and rmax * 4 > prev:
            break  # refinement stalled
        prev = rmax
        scale = float(rmax)
        rf = np.array([float(v) for v in resid]) / scale
        step = dv * (R.T @ lu.solve(rf)) * scale
        if not np.isfinite(step).all():
            break
        for t, v in enumerate(step.tolist()):
            if v:
                j = cols[t]
                d[j] = d.get(j, ZERO) + to_q(v)
        resid = _residual(rows, rhs, d)
    check_deadline(lim.deadline)
    if not any(resid):
        return d
    # Exact basic solution for what is left; prefer heavy columns as pivots.
    penalty = {j: int(-math.log2(weights[j])) // 8 for j in cols}
    sub = [{j: v for j, v in row.items() if j in pos} for row in rows]
    ncols = max((max(r) for r in sub if r), default=-1) + 1
    fix = solve_sparse(
        sub, resid, ncols, col_penalty=penalty, budget=lim.ops, deadline=lim.deadline
    )
    if fix is None:
        return None
    for j, v in enumerate(fix):
        if v:
            d[j] = d.get(j, ZERO) + v
    return d


def _residual(rows: Sequence[dict[int, Any]], rhs: Sequence[Any], d: dict[int, Any]) -> list:
    out = []
    for row, b in zip(rows, rhs, strict=True):
        t = b
        for j, v in row.items():
            dj = d.get(j)
            if dj:
                t -= v * dj
        out.append(t)
    return out


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
    fd = problem.float_data()
    s = fd.b - fd.A @ xf
    rows = [dict(zip(*A.row(i), strict=True)) for i in zero_rows]
    # Weighted least-norm correction: a column moves in proportion to the (relative) room
    # of the tightest inequality row it appears in, so near-active rows barely move.
    room = _room(problem, s)
    scale = 1.0 + float(np.abs(fd.b).max(initial=0.0))
    colroom = np.full(problem.n, np.inf)
    Acsc = fd.A.tocsc()
    for j in range(problem.n):
        ks = Acsc.indices[Acsc.indptr[j] : Acsc.indptr[j + 1]]
        if len(ks):
            colroom[j] = room[ks].min()
    weights = {j: min(1.0, max(1e-8, r / scale)) for j, r in enumerate(colroom.tolist()) if r > 0}
    try:
        d_map = weighted_min_norm(rows, resid, weights, lim)
    except BudgetExceeded:
        d_map = None
    if d_map is not None:
        return [xq[j] + d_map[j] if j in d_map else xq[j] for j in range(problem.n)]
    # Fallback: a basic solution, preferring columns away from near-active rows.
    tight = np.zeros(problem.n, dtype=bool)
    near = [i for i in other_rows if s[i] <= 1e-6 * (1.0 + abs(fd.b[i]))]
    if near:
        tight[np.unique(fd.A[near].indices)] = True
    penalty = {j: 1 for j in np.flatnonzero(tight).tolist()}
    d = solve_sparse(
        rows, resid, problem.n, col_penalty=penalty, budget=lim.ops, deadline=lim.deadline
    )
    if d is None:
        return None
    return [a + b if b else a for a, b in zip(xq, d, strict=True)]


def interior_primal(
    problem: ConicProblem, x: np.ndarray, eta: float, rounds: int = 6
) -> np.ndarray:
    """Float x whose slack is pushed into the interior of K by a relative margin eta.

    Each round gives the rows that lack room a target slack and solves
    ``min ||d|| s.t. A_R d = s_R - t_R`` (zero rows held at zero) by LSQR. Targets of
    different cones can conflict (a variable in two active cones), so rounds repeat on
    the rows still short of half the margin.
    """
    fd = problem.float_data()
    margin = eta * (1.0 + np.abs(fd.b))  # per-row relative margin
    zero_rows = np.array(_row_lists(problem)[0], dtype=np.int64)
    x = x.copy()
    for _ in range(rounds):
        s = fd.b - fd.A @ x
        target = s.copy()
        rows: list[int] = []
        for cone, a, b in problem.cone_slices():
            if cone.kind == "zero":
                target[a:b] = 0.0
                rows.extend(range(a, b))
            elif cone.kind == "nonneg":
                idx = np.flatnonzero(s[a:b] < 0.5 * margin[a:b]) + a
                target[idx] = margin[idx]
                rows.extend(idx.tolist())
            else:
                t, v = s[a], s[a + 1 : b]
                nv = float(np.linalg.norm(v))
                m = float(margin[a:b].max())
                if t - nv < 0.5 * m:
                    # Ask for the whole gap both by raising the head and by shrinking
                    # the tail: least squares meets at least half of each when one side
                    # is fixed (a head without variables, as in ||w - w0|| <= tau).
                    gap = nv + m - t
                    target[a] = t + gap
                    if nv > 0:
                        target[a + 1 : b] = v * max(0.0, 1.0 - gap / nv)
                    rows.extend(range(a, b))
        if len(rows) == sum(c.dim for c in problem.cones if c.kind == "zero"):
            break  # every cone row has room; only the equalities are targeted
        rows_arr = np.array(sorted(set(rows)), dtype=np.int64)
        # Equalities weigh 1e6 times more, so conflicts are resolved inside the cones.
        wt = np.where(np.isin(rows_arr, zero_rows), 1e6, 1.0)
        AR = sp.diags(wt) @ sp.csr_matrix(fd.A[rows_arr])
        rhs = wt * (s[rows_arr] - target[rows_arr])
        d = spla.lsqr(AR, rhs, atol=1e-15, btol=1e-15, iter_lim=20 * AR.shape[1] + 100)[0]
        x = x + d
    return x


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
    cols: list[dict[int, Any]] = [{} for _ in range(problem.n)]
    for k in range(problem.m):
        for p in range(A.indptr[k], A.indptr[k + 1]):
            cols[A.indices[p]][k] = A.data[p]
    # Weighted least-norm correction of the multipliers (weight: relative room in K*,
    # free rows weight 1), then a basic solution on rows with room as a fallback.
    zf = np.array([float(v) for v in z])
    room = _room(problem, zf)
    zscale = max(1.0, float(np.abs(zf).max(initial=0.0)))
    weights = {k: min(1.0, max(1e-8, r / zscale)) for k, r in enumerate(room.tolist()) if r > 0}
    try:
        d_map = weighted_min_norm(cols, [-r for r in rho], weights, lim)
    except BudgetExceeded:
        d_map = None
    if d_map is not None:
        return [z[k] + d_map[k] if k in d_map else z[k] for k in range(problem.m)]
    penalty: dict[int, int] = {}
    scale = max((abs(v) for v in z), default=ZERO)
    tiny = scale * Q(1, 10**6) if scale else Q(1, 10**12)
    for k in range(problem.m):
        if kinds[k] == "zero":
            continue
        penalty[k] = 0 if kinds[k] == "nonneg" and z[k] > tiny else (1 if kinds[k] == "soc" else 2)
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


def interior_dual(
    problem: ConicProblem, xq: list[Any], z: np.ndarray, eta: float, rounds: int = 6
) -> np.ndarray:
    """Float z pushed into int K* by a margin, then corrected toward ``A'z = -(Px + q)``
    by a step weighted by the room of each row (free rows move most).

    Row i gets margin ``eta (1 + ||q||) / (m (1 + |s_i|))``, so the pushes add at most
    ``eta (1 + ||q||)`` to the complementarity ``s'z``, but at least ten times the
    stationarity residual of z, which the exact correction later has to absorb. An SOC
    block short of room is pushed both by raising its head and by shrinking its tail,
    because stationarity can pin the head (as for the epigraph variable of a norm).
    Rounds repeat while some row is short of half its margin.
    """
    fd = problem.float_data()
    xf = np.array([int(v.numerator) / int(v.denominator) for v in xq])
    g = -(fd.P @ xf + fd.q)
    s = np.abs(fd.b - fd.A @ xf)
    margin = eta * (1.0 + np.abs(fd.q).max(initial=0.0)) / (max(problem.m, 1) * (1.0 + s))
    rho0 = float(np.abs(g - fd.A.T @ z).max(initial=0.0))
    margin = np.maximum(margin, 10.0 * rho0)
    u = z.copy()
    for _ in range(rounds):
        pushed = False
        w = np.ones_like(u)
        for cone, a, b in problem.cone_slices():
            if cone.kind == "nonneg":
                low = u[a:b] < 0.5 * margin[a:b]
                pushed |= bool(low.any())
                u[a:b] = np.where(low, margin[a:b], u[a:b])
                w[a:b] = u[a:b]
            elif cone.kind == "soc":
                nv = float(np.linalg.norm(u[a + 1 : b]))
                m = float(margin[a:b].min())
                if u[a] - nv < 0.5 * m:
                    pushed = True
                    gap = nv + m - u[a]
                    u[a] += gap
                    if nv > 0:
                        u[a + 1 : b] *= max(0.0, 1.0 - gap / nv)
                w[a:b] = u[a] - float(np.linalg.norm(u[a + 1 : b]))  # room
            else:
                w[a:b] = max(float(np.abs(u).max(initial=1.0)), 1.0)
        rho = g - fd.A.T @ u
        if not pushed and float(np.abs(rho).max(initial=0.0)) <= 0.1 * float(margin.min()):
            break
        if not np.any(rho):
            continue
        AtW = sp.csr_matrix(fd.A.T @ sp.diags(w))
        dd = spla.lsqr(AtW, rho, atol=1e-15, btol=1e-15, iter_lim=20 * AtW.shape[1] + 100)[0]
        u = u + w * dd
    return u


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
