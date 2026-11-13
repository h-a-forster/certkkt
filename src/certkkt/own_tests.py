"""Each solver's own stopping rule, evaluated exactly on the output it returned.

certkkt's certificate (DESIGN 3.2) applies one relative measure to every solver. A solver
that misses it has not necessarily broken its own promise: each solver stops on its own
residual definitions, norms and tolerances. This module re-implements those rules from
the solvers' source code, for the versions in ``uv.lock`` (DESIGN 3.5 cites the files), and
evaluates them in exact rational arithmetic on the data the solver was given (the
doubles in :class:`~certkkt.solvers.base.NativeRun`, each read as the rational it
denotes) and on the vectors it returned. Comparisons use the solver's own strict or
non-strict inequality. A criterion whose exact decision needs a quantity the solver does
not return is reported as undetermined (``passes is None``) unless the decision is the
same for every value that quantity can take.

Rules (unscaled quantities, solution normalised as returned; ``||.||`` is the inf-norm
unless stated):

- **OSQP 1.0.0** (``check_termination`` in ``src/auxil.c``, defaults ``eps_abs = eps_rel =
  1e-3``, ``scaled_termination = 0``, ``check_dualgap = 1``): ``||A x - z|| < eps_abs +
  eps_rel max(||z||, ||A x||)``; ``||P x + q + A'y|| < eps_abs + eps_rel max(||q||,
  ||A'y||, ||P x||)``; ``|x'P x + q'x + S(y)| < eps_abs + eps_rel max(|x'P x|, |q'x|,
  |S(y)|)`` with ``S(y) = u'max(y, 0) + l'min(y, 0)`` after projecting ``y`` onto the
  polar of the recession cone of ``[l, u]``. ``z`` is OSQP's internal projected iterate,
  which it does not return; all that is known is ``z`` in ``[l, u]``, so the primal test
  passes only if it passes for every such ``z`` and fails only if it fails for every one;
  otherwise it is undetermined, and OSQP's own float ``prim_res`` is kept as a diagnostic
  (see :func:`_osqp`). Status "solved inaccurate" uses 10x the tolerances.
- **SCS 3.3.1** (C source at commit cc244eb, bundled by scs-python 3.3.1;
  ``has_converged`` in ``src/scs.c``, defaults ``eps_abs = eps_rel = 1e-4``):
  ``||A x + s - b|| < eps_abs + eps_rel max(||b||, ||s||, ||A x||)``; ``||P x + A'y +
  c|| < eps_abs + eps_rel max(||c||, ||P x||, ||A'y||)``; ``|x'P x + c'x + b'y| <
  eps_abs + eps_rel max(|x'P x|, |c'x|, |b'y|)``.
- **Clarabel 0.11.1** (``info.rs``: ``update`` and ``is_solved``, defaults ``tol_feas =
  tol_gap_abs = tol_gap_rel = 1e-8``; "AlmostSolved" uses ``reduced_tol_*``): with
  2-norms ``|.|`` and inf-norms of the data, ``|A x + s - b| / max(1, ||b|| + |x| + |s|) <
  tol_feas``; ``|P x + q + A'z| / max(1, ||q|| + |x| + |z|) < tol_feas``; with ``p = q'x +
  x'P x / 2`` and ``d = -b'z - x'P x / 2``, ``|p - d| < tol_gap_abs`` or ``|p - d| / max(1,
  min(|p|, |d|)) < tol_gap_rel``. Its presolve drops nonnegative rows with ``b`` at
  infinity (``1e20``) first. The fourth condition, ``kappa / tau <= 1``, is internal to
  the homogeneous embedding and is not evaluated.
- **HiGHS 1.15.1, QP path**, in two stages. (1) The active-set solver
  (``highs/qpsolver/quass.cpp``) declares optimal only at a point it finds stationary on
  its working set (search direction ``||p||_2 < 1e-11``, or a zero step) where pricing
  finds no multiplier to release beyond ``dual_feasibility_tolerance``, for its internal
  Hessian ``P + qp_regularization_value I`` (``1e-7``). That needs the working set and is
  not evaluated. (2) ``Highs::checkOptimality`` and ``getKktFailures``
  (``HighsSolution.cpp``, ``primal_feasibility_tolerance = dual_feasibility_tolerance =
  1e-7``) then keep "Optimal" only if no column value or row activity violates a bound by
  more than the primal tolerance and no column or row dual has the wrong sign by more than
  the dual tolerance (the side is decided by which half of the bound interval the value
  lies in); otherwise the status becomes a solve error. This gate is what is evaluated, so
  a returned "Optimal" passes it in floats by construction and the exact evaluation tests
  only rounding. Row activities are taken as ``A x`` evaluated exactly; the activities
  HiGHS returned are tested too and reported as a diagnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from certkkt.rational import ONE, ZERO, Q, dot, max_abs, sqrt_down, sqrt_up, to_q
from certkkt.solvers.base import NativeRun, SolverResult
from certkkt.sparse import QMatrix

#: Bits of the rational square-root bounds used for 2-norms (Clarabel).
SQRT_BITS = 256

RULES = {
    "osqp": "OSQP 1.0.0 check_termination (src/auxil.c), unscaled",
    "scs": "SCS 3.3.1 @cc244eb has_converged (src/scs.c)",
    "clarabel": "Clarabel 0.11.1 DefaultInfo::update + is_solved (info.rs)",
    "highs": "HiGHS 1.15.1 checkOptimality + getKktFailures (QP path)",
}


@dataclass
class Criterion:
    """One condition of a stopping rule: ``value`` against ``threshold`` (nearest floats
    of the exact quantities; for a 2-norm, of a rational bound within 2**-120 relative).
    ``passes`` is the exact decision, or None when it cannot be decided exactly."""

    name: str
    value: float
    threshold: float
    passes: bool | None
    note: str = ""

    @property
    def ratio(self) -> float:
        if self.threshold > 0:
            return self.value / self.threshold
        return 0.0 if self.value == 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "ratio": self.ratio,
            "passes": self.passes,
        }
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class OwnTest:
    solver: str
    rule: str
    #: which tolerances were applied: "full", or "reduced" for a claim of reduced accuracy
    level: str
    criteria: list[Criterion]
    #: conditions of the rule that this evaluation cannot check
    not_evaluated: list[str] = field(default_factory=list)
    #: the solver's own float report of the same quantities, where it gives one
    solver_report: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def passes(self) -> bool | None:
        """False if any criterion fails exactly, True if all pass, else None."""
        verdicts = [c.passes for c in self.criteria]
        if any(v is False for v in verdicts):
            return False
        if all(v is True for v in verdicts):
            return True
        return None

    @property
    def binding(self) -> str:
        """The criterion with the largest value/threshold ratio."""
        return max(self.criteria, key=lambda c: c.ratio).name if self.criteria else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "rule": self.rule,
            "level": self.level,
            "passes": self.passes,
            "binding": self.binding,
            "criteria": [c.to_dict() for c in self.criteria],
            "not_evaluated": self.not_evaluated,
            "solver_report": self.solver_report,
            "diagnostics": self.diagnostics,
        }


class OwnTestUnavailable(ValueError):
    """The solver's rule cannot be evaluated for this run (stated in the message)."""


# -- helpers -----------------------------------------------------------------------------


def _qvec(a: np.ndarray) -> list[Any]:
    return [to_q(float(v)) for v in np.asarray(a, dtype=np.float64).ravel()]


def _sym(tri: Any, n: int) -> QMatrix:
    """Exact symmetric matrix from the triangle a solver was given (it reads it as
    ``T + T' - diag(T)``)."""
    t = sp.coo_array(sp.csr_array(tri) if tri is not None else sp.csr_array((n, n)))
    off = t.row != t.col
    rows = np.concatenate([t.row, t.col[off]])
    cols = np.concatenate([t.col, t.row[off]])
    vals = np.concatenate([t.data, t.data[off]])
    return QMatrix.from_triplets((n, n), rows.tolist(), cols.tolist(), vals.tolist())


def _f(v: Any) -> float:
    """Nearest float of a rational (inf on overflow)."""
    try:
        return int(v.numerator) / int(v.denominator)
    except OverflowError:
        return float("inf") if v > 0 else float("-inf")


def _decide_lt(lhs: Any, rhs: Any) -> bool:
    return bool(lhs < rhs)


def _crit(name: str, value: Any, threshold: Any, passes: bool | None, note: str = "") -> Criterion:
    return Criterion(name, _f(value), _f(threshold), passes, note)


def _norm2_bounds(v: Sequence[Any]) -> tuple[Any, Any]:
    s = dot(v, v)
    return sqrt_down(s, SQRT_BITS), sqrt_up(s, SQRT_BITS)


def _interval_lt(lhs: tuple[Any, Any], rhs: tuple[Any, Any]) -> bool | None:
    """Exact decision of ``l < r`` from enclosures ``l in lhs``, ``r in rhs``."""
    if lhs[1] < rhs[0]:
        return True
    if lhs[0] >= rhs[1]:
        return False
    return None


# -- OSQP ----------------------------------------------------------------------------------


def _osqp(nat: NativeRun, claim: str) -> OwnTest:
    st = nat.settings
    if st.get("scaling") and st.get("scaled_termination"):
        raise OwnTestUnavailable("scaled_termination needs OSQP's internal scaling")
    if st.get("polishing"):
        raise OwnTestUnavailable("polishing replaces the ADMM iterate; rule not modelled")
    approx = claim == "inaccurate"
    scale = Q(10) if approx else ONE
    eps_abs = to_q(float(st["eps_abs"])) * scale
    eps_rel = to_q(float(st["eps_rel"])) * scale
    d = nat.data
    n = len(d["q"])
    P = _sym(d["P"], n)
    A = QMatrix.from_scipy(d["A"])
    q = _qvec(d["q"])
    inf = float(d["infinity"])
    lo_f, up_f = np.asarray(d["l"], dtype=np.float64), np.asarray(d["u"], dtype=np.float64)
    x = _qvec(nat.output["x"])
    y = _qvec(nat.output["y"])
    Px = P.matvec(x)
    Ax = A.matvec(x)
    Aty = A.rmatvec(y)
    criteria = []

    # dual residual
    dr = [a + b + c for a, b, c in zip(q, Px, Aty, strict=True)]
    dual_res = max_abs(dr)
    eps_dual = eps_abs + eps_rel * max(max_abs(q), max_abs(Aty), max_abs(Px))
    criteria.append(_crit("dual residual", dual_res, eps_dual, _decide_lt(dual_res, eps_dual)))

    # primal residual over every internal z in [l, u]. (The sign of y does not pin z to a
    # bound: OSQP updates y by a sum, so y_i can be a rounding residue where z_i is
    # interior.) The threshold is at least eps_abs + eps_rel ||A x|| for every z. It
    # passes for every z if even the farthest z in [l, u] (r_max) beats that bound. z0 =
    # proj(A x) minimises every |A x - z|_i, so r(z) >= r0 for all z, and ||z|| <= ||A x||
    # + r(z): if r0 (1 - eps_rel) >= eps_abs + eps_rel ||A x||, no z passes. In between
    # the exact decision depends on z and is undetermined.
    lo_inf = lo_f <= -inf
    up_inf = up_f >= inf
    lo = [None if li else to_q(float(v)) for v, li in zip(lo_f, lo_inf, strict=True)]
    up = [None if ui else to_q(float(v)) for v, ui in zip(up_f, up_inf, strict=True)]
    lo_c = [to_q(-inf) if v is None else v for v in lo]  # OSQP holds infinite bounds as 1e30
    up_c = [to_q(inf) if v is None else v for v in up]
    diagnostics: dict[str, Any] = {}
    if A.shape[0]:
        z0 = [min(max(ax, lc), uc) for ax, lc, uc in zip(Ax, lo_c, up_c, strict=True)]
        r0 = max_abs([a - z for a, z in zip(Ax, z0, strict=True)])
        r_max = max(
            (max(abs(a - lc), abs(a - uc)) for a, lc, uc in zip(Ax, lo_c, up_c, strict=True)),
            default=ZERO,
        )
        nAx = max_abs(Ax)
        thr_low = eps_abs + eps_rel * nAx
        if r_max < thr_low:
            verdict: bool | None = True
        elif eps_rel < 1 and r0 * (1 - eps_rel) >= thr_low:
            verdict = False
        else:
            verdict = None
        criteria.append(
            _crit(
                "primal residual",
                r0,
                eps_abs + eps_rel * max(max_abs(z0), nAx),
                verdict,
                "z unknown: value and threshold at z = proj(A x) (a lower bound on the "
                "value); passes only if every z in [l, u] passes, fails only if every z fails",
            )
        )
        if verdict is None and "prim_res" in nat.info:
            # OSQP's own float report of r(z) for its z, against the lower bound of the
            # threshold: a pass here holds whatever ||z|| is.
            rep = to_q(abs(float(nat.info["prim_res"])))
            diagnostics["undetermined_by_solver_report"] = {
                "primal residual": {
                    "reported": _f(rep),
                    "threshold_lower_bound": _f(thr_low),
                    "passes": bool(rep < thr_low),
                }
            }

    # duality gap
    if st.get("check_dualgap"):
        # OSQP treats a scaled bound beyond 1e26 as infinite; with scaling factors in
        # [1e-4, 1e4] a finite bound above 1e22 in magnitude is ambiguous.
        big = [
            i
            for i in range(len(lo))
            if (lo[i] is not None and abs(lo[i]) > Q(10) ** 22)
            or (up[i] is not None and abs(up[i]) > Q(10) ** 22)
        ]
        sc = ZERO
        for i, yi in enumerate(y):
            if yi > 0 and up[i] is not None:
                sc += up[i] * yi
            elif yi < 0 and lo[i] is not None:
                sc += lo[i] * yi
        xPx = dot(x, Px)
        qx = dot(q, x)
        gap = xPx + qx + sc
        eps_gap = eps_abs + eps_rel * max(abs(xPx), abs(qx), abs(sc))
        criteria.append(
            _crit(
                "duality gap",
                abs(gap),
                eps_gap,
                None if big else _decide_lt(abs(gap), eps_gap),
                "finite bounds above 1e22 make the infinite-bound test ambiguous" if big else "",
            )
        )
    return OwnTest(
        "osqp",
        RULES["osqp"],
        "reduced (10x)" if approx else "full",
        criteria,
        not_evaluated=[
            "OSQP rounds |y_i| <= 1e-15 (scaled) to zero inside the gap's support function"
        ],
        solver_report=dict(nat.info),
        diagnostics=diagnostics,
    )


# -- SCS -----------------------------------------------------------------------------------


def _scs(nat: NativeRun, claim: str) -> OwnTest:
    st = nat.settings
    eps_abs, eps_rel = to_q(float(st["eps_abs"])), to_q(float(st["eps_rel"]))
    d = nat.data
    n = len(d["q"])
    P = _sym(d["P"], n)
    A = QMatrix.from_scipy(d["A"])
    c = _qvec(d["q"])
    b = _qvec(d["b"])
    x, y, s = (_qvec(nat.output[k]) for k in ("x", "y", "s"))
    Ax, Px, Aty = A.matvec(x), P.matvec(x), A.rmatvec(y)
    rp = max_abs([a + si - bi for a, si, bi in zip(Ax, s, b, strict=True)])
    prl = max(max_abs(b), max_abs(s), max_abs(Ax))
    rd = max_abs([a + t + ci for a, t, ci in zip(Px, Aty, c, strict=True)])
    drl = max(max_abs(c), max_abs(Px), max_abs(Aty))
    xPx, cx, by = dot(x, Px), dot(c, x), dot(b, y)
    gap = abs(xPx + cx + by)
    grl = max(abs(xPx), abs(cx), abs(by))
    crit = [
        ("primal residual", rp, eps_abs + eps_rel * prl),
        ("dual residual", rd, eps_abs + eps_rel * drl),
        ("duality gap", gap, eps_abs + eps_rel * grl),
    ]
    return OwnTest(
        "scs",
        RULES["scs"],
        "full",
        [_crit(nm, v, t, _decide_lt(v, t)) for nm, v, t in crit],
        not_evaluated=[] if claim == "optimal" else ["SCS states no rule for inaccurate output"],
        solver_report=dict(nat.info),
    )


# -- Clarabel ------------------------------------------------------------------------------


def _clarabel(nat: NativeRun, claim: str) -> OwnTest:
    st = nat.settings
    reduced = claim == "inaccurate"
    pre = "reduced_" if reduced else ""
    tol_feas = to_q(float(st[pre + "tol_feas"]))
    tol_abs = to_q(float(st[pre + "tol_gap_abs"]))
    tol_rel = to_q(float(st[pre + "tol_gap_rel"]))
    d = nat.data
    n = len(d["q"])
    P = _sym(d["P"], n)
    A_s = sp.csr_array(d["A"])
    b_f = np.asarray(d["b"], dtype=np.float64)
    infb = float(st["infinity"])
    keep = np.ones(len(b_f), dtype=bool)
    if st.get("presolve_enable"):
        row = 0
        cut = (1.0 - np.finfo(np.float64).eps * 10.0) * infb
        for kind, dim in d["cones"]:
            if kind == "nonneg":
                keep[row : row + dim] = ~(b_f[row : row + dim] > cut)
            row += dim
    b_f = np.minimum(b_f[keep], infb)
    A = QMatrix.from_scipy(A_s[keep] if not keep.all() else A_s)
    b = _qvec(b_f)
    q = _qvec(d["q"])
    x = _qvec(nat.output["x"])
    z = _qvec(nat.output["z"][keep])
    s = _qvec(nat.output["s"][keep])
    Ax, Px, Atz = A.matvec(x), P.matvec(x), A.rmatvec(z)
    rp = [a + si - bi for a, si, bi in zip(Ax, s, b, strict=True)]
    rd = [a + t + qi for a, t, qi in zip(Px, Atz, q, strict=True)]
    nx, ns, nz = _norm2_bounds(x), _norm2_bounds(s), _norm2_bounds(z)
    nb, nq = max_abs(b), max_abs(q)

    def ratio_test(name: str, r: list[Any], na: Any, n1: tuple, n2: tuple) -> Criterion:
        num = _norm2_bounds(r)
        den = (max(ONE, na + n1[0] + n2[0]), max(ONE, na + n1[1] + n2[1]))
        # num / den < tol  <=>  num < tol * den; the value is an upper bound of the ratio
        verdict = _interval_lt(num, (tol_feas * den[0], tol_feas * den[1]))
        return Criterion(name, _f(num[1] / den[0]), _f(tol_feas), verdict)

    cp_ = ratio_test("primal residual", rp, nb, nx, ns)
    cd_ = ratio_test("dual residual", rd, nq, nx, nz)
    xPx = dot(x, Px)
    cost_p = dot(q, x) + xPx / 2
    cost_d = -dot(b, z) - xPx / 2
    gap_abs = abs(cost_p - cost_d)
    den = max(ONE, min(abs(cost_p), abs(cost_d)))
    gap_thr = max(tol_abs, tol_rel * den)  # |g| < tol_abs or |g| / den < tol_rel
    cg = _crit(
        "duality gap",
        gap_abs,
        gap_thr,
        _decide_lt(gap_abs, gap_thr),
        "absolute or relative: threshold max(tol_gap_abs, tol_gap_rel * max(1, min(|p|, |d|)))",
    )
    return OwnTest(
        "clarabel",
        RULES["clarabel"],
        "reduced" if reduced else "full",
        [cp_, cd_, cg],
        not_evaluated=["kappa/tau <= 1 (internal to the homogeneous embedding)"],
        solver_report=dict(nat.info),
        diagnostics={"presolve_removed_rows": int((~keep).sum())},
    )


# -- HiGHS ---------------------------------------------------------------------------------


def _highs_failures(
    lower: list[Any | None],
    upper: list[Any | None],
    value: list[Any],
    dual: list[Any],
    tol_p: Any,
) -> tuple[Any, Any]:
    """Max primal bound violation and max dual sign infeasibility (getVariableKktFailures).
    ``None`` marks an infinite bound."""
    max_p, max_d = ZERO, ZERO
    for lo, up, v, du in zip(lower, upper, value, dual, strict=True):
        if lo is not None and v < lo - tol_p:
            max_p = max(max_p, lo - v)
        if up is not None and v > up + tol_p:
            max_p = max(max_p, v - up)
        if lo is None and up is None:
            di = abs(du)
        elif lo is not None and up is not None and (up <= lo or (up - lo) ** 2 <= tol_p):
            di = ZERO  # fixed, or an interval too short to tell its sides apart
        else:
            below_middle = up is None or (lo is not None and v < (lo + up) / 2)
            di = max(-du, ZERO) if below_middle else max(du, ZERO)
        max_d = max(max_d, di)
    return max_p, max_d


def _highs(nat: NativeRun, claim: str) -> OwnTest:
    st = nat.settings
    d = nat.data
    if not nat.output:
        raise OwnTestUnavailable("HiGHS returned no primal-dual solution")
    if d["P"].nnz == 0:
        raise OwnTestUnavailable("LP path (simplex or IPM) not modelled; only the QP path is")
    if st.get("solver") not in ("choose", "qpasm"):
        raise OwnTestUnavailable(f"solver={st.get('solver')!r}: only the active-set QP path")
    tol_p = to_q(float(st["primal_feasibility_tolerance"]))
    tol_d = to_q(float(st["dual_feasibility_tolerance"]))
    if float(st["kkt_tolerance"]) != 1e-7:  # kDefaultKktTolerance: an override covers both
        tol_p = tol_d = to_q(float(st["kkt_tolerance"]))

    def bounds(a: np.ndarray) -> list[Any | None]:
        return [None if not np.isfinite(v) else to_q(float(v)) for v in a]

    A = QMatrix.from_scipy(d["A"])
    x = _qvec(nat.output["col_value"])
    Ax = A.matvec(x)
    row_rep = _qvec(nat.output["row_value"])
    col_dual, row_dual = _qvec(nat.output["col_dual"]), _qvec(nat.output["row_dual"])
    cl, cu = bounds(d["col_lower"]), bounds(d["col_upper"])
    rl, ru = bounds(d["row_lower"]), bounds(d["row_upper"])
    pc, dc = _highs_failures(cl, cu, x, col_dual, tol_p)
    pr, dr = _highs_failures(rl, ru, Ax, row_dual, tol_p)
    pr_rep, dr_rep = _highs_failures(rl, ru, row_rep, row_dual, tol_p)
    max_p, max_d = max(pc, pr), max(dc, dr)
    criteria = [
        _crit("primal feasibility", max_p, tol_p, bool(max_p <= tol_p)),
        _crit("dual feasibility", max_d, tol_d, bool(max_d <= tol_d)),
    ]
    rep_p, rep_d = max(pc, pr_rep), max(dc, dr_rep)
    return OwnTest(
        "highs",
        RULES["highs"],
        "full",
        criteria,
        not_evaluated=[
            "the active-set solver's own stop (zero search direction on its working set, "
            "no multiplier to release) on its regularised Hessian: needs the working set"
        ],
        solver_report=dict(nat.info),
        diagnostics={
            "with_reported_row_values": {
                "max_primal_infeasibility": _f(rep_p),
                "max_dual_infeasibility": _f(rep_d),
                "passes": bool(rep_p <= tol_p and rep_d <= tol_d),
            },
            "max_row_value_error": _f(max_abs([a - r for a, r in zip(Ax, row_rep, strict=True)])),
        },
    )


_EVALUATORS = {"osqp": _osqp, "scs": _scs, "clarabel": _clarabel, "highs": _highs}


def own_test(result: SolverResult) -> OwnTest:
    """Evaluate the solver's own stopping rule exactly on its returned output.

    The tolerances are those of the status claimed: the full ones for "optimal", the
    reduced ones for "inaccurate" where the solver defines them (Clarabel, OSQP). Raises
    :class:`OwnTestUnavailable` when the rule cannot be evaluated for this run.
    """
    nat = result.native
    if nat is None:
        raise OwnTestUnavailable(f"{result.solver}: the adapter kept no native run")
    fn = _EVALUATORS.get(result.solver)
    if fn is None:
        raise OwnTestUnavailable(f"{result.solver}: stopping rule not implemented")
    if not result.has_solution:
        raise OwnTestUnavailable("no solution returned")
    for k, v in nat.output.items():
        if v is None or not np.isfinite(v).all():
            raise OwnTestUnavailable(f"output {k!r} missing or not finite")
    return fn(nat, result.status)


__all__ = ["RULES", "Criterion", "OwnTest", "OwnTestUnavailable", "own_test"]
