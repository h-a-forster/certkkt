"""Exact certification of a primal-dual candidate for a canonical conic problem.

Given exact data and a rational candidate ``(x, z)``, the slack is recomputed exactly as
``s = b - A x`` (never taken from the solver) and every quantity below is evaluated in
exact rational arithmetic:

* primal: equality residual ``max |s_i|`` over zero-cone rows and the cone violation of
  ``s`` (``max(0, -s_i)`` for nonneg rows, ``max(0, ||s_1:|| - s_0)`` for SOC blocks);
* dual: stationarity residual ``||P x + q + A'z||_inf`` and the dual cone violation of z;
* objectives ``pobj = 1/2 x'Px + q'x + r`` and ``dobj = -1/2 x'Px - b'z + r``, the gap
  ``pobj - dobj`` and the complementarity ``s'z``.

SOC norms are irrational in general; their violations are reported as rational upper
bounds (relative slack below 2**-60), while exact membership is decided without square
roots (``s_0 >= 0`` and ``s_0^2 >= ||s_1:||^2``).

Relative measures (DESIGN section 3): ``rel_primal = r_P / (1 + ||b||_inf)``,
``rel_dual = r_D / (1 + ||q||_inf)``, ``rel_gap = |gap| / (1 + |pobj| + |dobj|)``.

Certificate levels, strongest first (DESIGN section 3.3):

* ``optimal``: s in K and z in K* exactly, stationarity exact, gap exactly 0;
* ``bounded``: exactly feasible primal and dual, gap > 0, so the optimal value lies in
  ``[dobj, pobj]``;
* ``primal_feasible`` / ``dual_feasible``: one side exactly feasible (an exact upper or
  lower bound on the optimal value);
* ``residuals``: neither side exactly feasible; only the exact residuals are reported.

A lower bound from the dual requires P to be positive semidefinite. Dual-based levels are
granted only when P is zero or its semidefiniteness was verified exactly; otherwise the
level is downgraded and ``psd`` says why.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from certkkt.problem import ConicProblem
from certkkt.rational import ONE, ZERO, Q, dot, max_abs, q_float_up, sqrt_up, to_q
from certkkt.sparse import DEFAULT_BUDGET, psd_check

LEVELS = ("optimal", "bounded", "primal_feasible", "dual_feasible", "residuals")
#: Levels each level implies. The order is partial: primal_feasible and dual_feasible are
#: incomparable, so neither satisfies a requirement for the other.
_IMPLIES = {
    "optimal": frozenset(LEVELS),
    "bounded": frozenset(("bounded", "primal_feasible", "dual_feasible", "residuals")),
    "primal_feasible": frozenset(("primal_feasible", "residuals")),
    "dual_feasible": frozenset(("dual_feasible", "residuals")),
    "residuals": frozenset(("residuals",)),
}


def meets_level(level: str, required: str) -> bool:
    """True if a certificate at ``level`` is at least as strong as ``required``."""
    if required not in _IMPLIES:
        raise ValueError(f"unknown level {required!r}; choose from {LEVELS}")
    return required in _IMPLIES[level]


#: Thresholds at which certificates are classified by default (exact decimals).
DEFAULT_EPS = ("1e-4", "1e-6", "1e-8", "1e-10")


@dataclass
class Certificate:
    """Exact certification outcome. Rational fields are exact or rigorous upper bounds."""

    method: str
    level: str
    psd: str
    has_dual: bool
    primal_eq_residual: Any
    primal_cone_violation: Any
    primal_member: bool
    stationarity_residual: Any | None
    dual_cone_violation: Any | None
    dual_member: bool
    pobj: Any
    dobj: Any | None
    gap: Any | None
    complementarity: Any | None
    rel_primal: Any
    rel_dual: Any | None
    rel_gap: Any | None
    notes: list[str] = field(default_factory=list)

    @property
    def primal_residual(self) -> Any:
        return max(self.primal_eq_residual, self.primal_cone_violation)

    @property
    def dual_residual(self) -> Any | None:
        if self.stationarity_residual is None or self.dual_cone_violation is None:
            return None
        return max(self.stationarity_residual, self.dual_cone_violation)

    @property
    def max_rel(self) -> Any | None:
        """The smallest eps at which the candidate is certified (None without a dual)."""
        if self.rel_dual is None or self.rel_gap is None:
            return None
        return max(self.rel_primal, self.rel_dual, self.rel_gap)

    def certified_at(self, eps: str | float | Any) -> bool:
        """True if every relative residual is at most eps (exact comparison)."""
        mr = self.max_rel
        return mr is not None and mr <= to_q(eps)

    def rank(self) -> tuple[int, float, int]:
        """Sort key, lower is better: optimal, then bounded, then everything else by the
        largest relative residual (a one-sided exact certificate with a huge gap is valid
        but says less than a near-feasible pair), ties broken by level."""
        mr = self.max_rel
        tier = {"optimal": 0, "bounded": 1}.get(self.level, 2)
        level = LEVELS.index(self.level)
        return (tier, q_float_up(mr) if mr is not None else math.inf, level)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: rationals become floats rounded up (rigorous upper bounds),
        objectives become nearest floats, and exact values are kept as strings when short.
        """

        def up(v: Any | None) -> float | None:
            return None if v is None else q_float_up(v)

        def near(v: Any | None) -> float | None:
            return None if v is None else int(v.numerator) / int(v.denominator)

        return {
            "method": self.method,
            "level": self.level,
            "psd": self.psd,
            "has_dual": self.has_dual,
            "primal_eq_residual": up(self.primal_eq_residual),
            "primal_cone_violation": up(self.primal_cone_violation),
            "primal_member": self.primal_member,
            "stationarity_residual": up(self.stationarity_residual),
            "dual_cone_violation": up(self.dual_cone_violation),
            "dual_member": self.dual_member,
            "pobj": near(self.pobj),
            "dobj": near(self.dobj),
            "gap": near(self.gap),
            "gap_up": up(self.gap),
            "complementarity": near(self.complementarity),
            "rel_primal": up(self.rel_primal),
            "rel_dual": up(self.rel_dual),
            "rel_gap": up(self.rel_gap),
            "max_rel": up(self.max_rel),
            "exact_gap": _short(self.gap),
            "notes": list(self.notes),
        }


def _short(v: Any | None, limit: int = 200) -> str | None:
    if v is None:
        return None
    num, den = int(v.numerator), int(v.denominator)
    # Skip the decimal conversion of huge integers (slow, and capped by Python at 4300
    # digits); 4 bits per digit over-approximates the digit count.
    if num.bit_length() + den.bit_length() > 4 * limit:
        return None
    s = f"{num}/{den}" if den != 1 else str(num)
    return s if len(s) <= limit else None


def psd_status(problem: ConicProblem, budget: int = DEFAULT_BUDGET) -> str:
    """ "zero", "psd", "not_psd" or "unknown" (budget exhausted), cached in problem.meta."""
    status = problem.meta.get("psd_status")
    if status is None:
        status = "zero" if problem.P.nnz == 0 else psd_check(problem.P, budget=budget).status
        problem.meta["psd_status"] = status
    return status


def cone_violation(problem: ConicProblem, v: Sequence[Any], dual: bool) -> tuple[Any, bool]:
    """(rigorous upper bound on the cone violation, exact membership) for v in K or K*.

    For the primal, zero-cone rows count as violation ``|v_i|`` here too; callers that
    report the equality residual separately use :func:`primal_parts`.
    """
    worst = ZERO
    member = True
    for cone, a, b in problem.cone_slices():
        if cone.kind == "zero":
            if dual:
                continue
            for i in range(a, b):
                av = abs(v[i])
                if av:
                    member = False
                    worst = max(worst, av)
        elif cone.kind == "nonneg":
            for i in range(a, b):
                if v[i] < 0:
                    member = False
                    worst = max(worst, -v[i])
        else:  # soc
            t = v[a]
            sq = ZERO
            for i in range(a + 1, b):
                if v[i]:
                    sq += v[i] * v[i]
            if t < 0 or t * t < sq:
                member = False
                worst = max(worst, sqrt_up(sq) - t)
    return worst, member


def primal_parts(problem: ConicProblem, s: Sequence[Any]) -> tuple[Any, Any, bool]:
    """(equality residual, inequality-cone violation upper bound, exact membership)."""
    eq = ZERO
    for cone, a, b in problem.cone_slices():
        if cone.kind == "zero":
            for i in range(a, b):
                av = abs(s[i])
                if av > eq:
                    eq = av
    viol = ZERO
    member = eq == 0
    for cone, a, b in problem.cone_slices():
        if cone.kind == "nonneg":
            for i in range(a, b):
                if s[i] < 0:
                    member = False
                    if -s[i] > viol:
                        viol = -s[i]
        elif cone.kind == "soc":
            t = s[a]
            sq = ZERO
            for i in range(a + 1, b):
                if s[i]:
                    sq += s[i] * s[i]
            if t < 0 or t * t < sq:
                member = False
                viol = max(viol, sqrt_up(sq) - t)
    return eq, viol, member


def certify(
    problem: ConicProblem,
    x: Sequence[Any],
    z: Sequence[Any] | None,
    *,
    method: str = "as_is",
    psd_budget: int = DEFAULT_BUDGET,
) -> Certificate:
    """Certify a candidate. ``x`` and ``z`` may be floats (converted exactly) or rationals."""
    n, m = problem.n, problem.m
    xq = _exact(x, n, "x")
    zq = None if z is None else _exact(z, m, "z")
    ax = problem.A.matvec(xq)
    s = [bi - ai for bi, ai in zip(problem.b, ax, strict=True)]
    px = problem.P.matvec(xq)
    xpx = dot(xq, px)
    half = Q(1, 2)
    pobj = half * xpx + dot(problem.q, xq) + problem.r
    eq, viol, p_member = primal_parts(problem, s)
    b_scale = ONE + max_abs(problem.b)
    q_scale = ONE + max_abs(problem.q)
    rel_primal = max(eq, viol) / b_scale
    notes: list[str] = []

    if zq is None:
        stat = dviol = dobj = gap = comp = rel_dual = rel_gap = None
        d_member = False
    else:
        atz = problem.A.rmatvec(zq)
        rho = [pi + qi + ai for pi, qi, ai in zip(px, problem.q, atz, strict=True)]
        stat = max_abs(rho)
        dviol, d_member = cone_violation(problem, zq, dual=True)
        d_member = d_member and stat == 0
        dobj = -half * xpx - dot(problem.b, zq) + problem.r
        gap = pobj - dobj
        comp = dot(s, zq)
        rel_dual = max(stat, dviol) / q_scale
        rel_gap = abs(gap) / (ONE + abs(pobj) + abs(dobj))

    psd = psd_status(problem, psd_budget)
    lower_ok = psd in ("zero", "psd")
    if p_member and d_member:
        level = "optimal" if gap == 0 else "bounded"
        if gap is not None and gap < 0:  # impossible for exactly feasible pairs
            raise AssertionError("negative gap for an exactly feasible pair")
    elif p_member:
        level = "primal_feasible"
    elif d_member:
        level = "dual_feasible"
    else:
        level = "residuals"
    if level in ("optimal", "bounded", "dual_feasible") and not lower_ok:
        notes.append(f"dual bound not valid: P semidefiniteness is {psd}")
        level = "primal_feasible" if p_member else "residuals"

    return Certificate(
        method=method,
        level=level,
        psd=psd,
        has_dual=zq is not None,
        primal_eq_residual=eq,
        primal_cone_violation=viol,
        primal_member=p_member,
        stationarity_residual=stat,
        dual_cone_violation=dviol,
        dual_member=d_member,
        pobj=pobj,
        dobj=dobj,
        gap=gap,
        complementarity=comp,
        rel_primal=rel_primal,
        rel_dual=rel_dual,
        rel_gap=rel_gap,
        notes=notes,
    )


def _exact(v: Sequence[Any], length: int, name: str) -> list[Any]:
    if isinstance(v, np.ndarray):
        if v.shape != (length,):
            raise ValueError(f"{name} has shape {v.shape}, expected ({length},)")
        if not np.isfinite(v).all():
            raise ValueError(f"{name} has non-finite entries")
        return [to_q(f) for f in v.tolist()]
    if len(v) != length:
        raise ValueError(f"{name} has length {len(v)}, expected {length}")
    return [to_q(f) for f in v]


def float_residuals(problem: ConicProblem, x: np.ndarray, z: np.ndarray | None) -> dict:
    """The same relative measures evaluated naively in float64 (for comparison only)."""
    fd = problem.float_data()
    s = fd.b - fd.A @ x
    eq = 0.0
    viol = 0.0
    for cone, a, b in problem.cone_slices():
        blk = s[a:b]
        if cone.kind == "zero":
            eq = max(eq, float(np.max(np.abs(blk))))
        elif cone.kind == "nonneg":
            viol = max(viol, float(np.max(-blk, initial=0.0)))
        else:
            viol = max(viol, float(np.linalg.norm(blk[1:]) - blk[0]), 0.0)
    px = fd.P @ x
    pobj = 0.5 * float(x @ px) + float(fd.q @ x) + fd.r
    out: dict[str, float | None] = {
        "rel_primal": max(eq, viol) / (1.0 + float(np.max(np.abs(fd.b), initial=0.0))),
        "rel_dual": None,
        "rel_gap": None,
    }
    if z is not None:
        rho = px + fd.q + fd.A.T @ z
        dv = 0.0
        for cone, a, b in problem.cone_slices():
            blk = z[a:b]
            if cone.kind == "nonneg":
                dv = max(dv, float(np.max(-blk, initial=0.0)))
            elif cone.kind == "soc":
                dv = max(dv, float(np.linalg.norm(blk[1:]) - blk[0]), 0.0)
        dobj = -0.5 * float(x @ px) - float(fd.b @ z) + fd.r
        out["rel_dual"] = max(float(np.max(np.abs(rho), initial=0.0)), dv) / (
            1.0 + float(np.max(np.abs(fd.q), initial=0.0))
        )
        out["rel_gap"] = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
    return out


__all__ = [
    "DEFAULT_EPS",
    "LEVELS",
    "meets_level",
    "Certificate",
    "certify",
    "cone_violation",
    "float_residuals",
    "primal_parts",
    "psd_status",
]
