"""Aggregate a study's units.jsonl into the tables of REPORT.md (DESIGN section 6)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from certkkt.stats import fmt_share, quantile, wilson

EPS = ("1e-4", "1e-6", "1e-8")
EXACT_LEVELS = ("optimal", "bounded")


def _f(e: str) -> float:
    return float(e)


def asis_rel(rec: dict[str, Any]) -> float | None:
    """Exact (rounded-up) largest relative residual of the solver's floats, or None."""
    a = rec.get("as_is")
    return None if not a else a.get("max_rel")


def asis_ok(rec: dict[str, Any], eps: str) -> bool:
    v = asis_rel(rec)
    return v is not None and v <= _f(eps)


def repaired_ok(rec: dict[str, Any], eps: str) -> bool:
    """An exactly feasible primal-dual pair whose exact relative gap is at most eps."""
    b = rec.get("best")
    if not b or b.get("level") not in EXACT_LEVELS:
        return False
    return b.get("max_rel") is not None and b["max_rel"] <= _f(eps)


def exactly_optimal(rec: dict[str, Any]) -> bool:
    b = rec.get("best")
    return bool(b) and b.get("level") == "optimal"


def float_disagrees(rec: dict[str, Any], eps: str) -> bool:
    """Float64 evaluation of the residuals and exact evaluation disagree at eps."""
    fl = rec.get("float")
    a = asis_rel(rec)
    if not fl or a is None or fl.get("rel_dual") is None:
        return False
    fv = max(fl["rel_primal"], fl["rel_dual"], fl["rel_gap"])
    return (fv <= _f(eps)) != (a <= _f(eps))


def _group(records: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[tuple(r.get(k) for k in keys)].append(r)
    return dict(sorted(out.items()))


def _status(rec: dict[str, Any]) -> str:
    return rec.get("result", {}).get("status") or rec.get("outcome", "?")


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Machine-readable summary: counts and Wilson intervals per suite/solver/tolerance."""
    out: dict[str, Any] = {}
    for (suite, solver, tol), recs in _group(records, "suite", "solver", "tol").items():
        n = len(recs)
        row: dict[str, Any] = {
            "n": n,
            "claimed_optimal": sum(_status(r) == "optimal" for r in recs),
        }
        for eps in EPS:
            for name, pred in (("as_is", asis_ok), ("repaired", repaired_ok)):
                k = sum(pred(r, eps) for r in recs)
                lo, hi = wilson(k, n)
                row[f"{name}@{eps}"] = {"k": k, "p": k / n if n else None, "ci95": [lo, hi]}
        k = sum(exactly_optimal(r) for r in recs)
        row["exactly_optimal"] = {"k": k, "ci95": list(wilson(k, n))}
        out[f"{suite}/{solver}/{tol}"] = row
    return out


def markdown(records: list[dict[str, Any]], title: str = "certkkt study") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"{len(records)} units. Shares are `k/n = p% [95% Wilson interval]`.")
    lines.append("")
    outcomes = defaultdict(int)
    for r in records:
        outcomes[r.get("outcome", "?")] += 1
    lines.append("Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())))
    lines.append("")

    default = [r for r in records if r.get("tol") == "default"]
    if default:
        lines += [
            "## Certification at solver defaults",
            "",
            "As-is: the solver's floats, converted exactly; every exact relative residual",
            "(primal, dual, gap) at most eps. Repaired: an exactly feasible primal-dual",
            "pair (level optimal or bounded) with exact relative gap at most eps.",
            "",
            "| suite | solver | n | claims optimal | as-is 1e-4 | as-is 1e-6 | as-is 1e-8 "
            "| repaired 1e-6 | repaired 1e-8 | exact optimum |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (suite, solver), recs in _group(default, "suite", "solver").items():
            n = len(recs)
            cells = [
                suite,
                solver,
                str(n),
                fmt_share(sum(_status(r) == "optimal" for r in recs), n),
                *(fmt_share(sum(asis_ok(r, e) for r in recs), n) for e in EPS),
                *(fmt_share(sum(repaired_ok(r, e) for r in recs), n) for e in ("1e-6", "1e-8")),
                fmt_share(sum(exactly_optimal(r) for r in recs), n),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        dis = [r for r in default if float_disagrees(r, "1e-8")]
        lines.append(
            f"Float64 and exact evaluation of the as-is residuals disagree at eps = 1e-8 on "
            f"{len(dis)} of {len(default)} default units"
            + (": " + ", ".join(r["key"] for r in dis[:10]) if dis else "")
            + ("" if len(dis) <= 10 else ", ...")
            + "."
        )
        lines.append("")

    swept = [r for r in records if r.get("tol") != "default"]
    if swept:
        lines += [
            "## Requested tolerance versus certified accuracy",
            "",
            "Median [quartiles] of log10 of the exact as-is relative residual (largest of",
            "primal, dual, gap) and of the exact relative gap alone, over units with a",
            "solution; `<= tol` is the share whose exact residual meets the requested",
            "tolerance under certkkt's relative measure (solvers scale differently).",
            "",
            "| suite | solver | tol | n | log10 max_rel | log10 rel_gap | max_rel <= tol "
            "| repaired at tol |",
            "|---|---|---|---|---|---|---|---|",
        ]
        tol_order = {t: i for i, t in enumerate(("default", "1e-4", "1e-6", "1e-8"))}
        groups = _group(records, "suite", "solver", "tol")
        for (suite, solver, tol), recs in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], tol_order.get(kv[0][2], 9))
        ):
            n = len(recs)
            mr = [math.log10(max(v, 1e-300)) for r in recs if (v := asis_rel(r)) is not None]
            gaps = [
                math.log10(max(r["as_is"]["rel_gap"], 1e-300))
                for r in recs
                if r.get("as_is") and r["as_is"].get("rel_gap") is not None
            ]
            if tol == "default":
                ok = rep = "-"
            else:
                ok = fmt_share(sum(asis_ok(r, tol) for r in recs), n)
                repaired = any("best" in r for r in recs)
                rep = fmt_share(sum(repaired_ok(r, tol) for r in recs), n) if repaired else "-"
            lines.append(
                f"| {suite} | {solver} | {tol} | {n} | {_q3(mr)} | {_q3(gaps)} | {ok} | {rep} |"
            )
        lines.append("")

    mitt = [r for r in records if r.get("suite") == "mittelmann"]
    if mitt:
        lines += [
            "## Mittelmann SOCP instances (downloadable subset)",
            "",
            "| instance | solver | status | solve s | as-is max_rel | best level |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(mitt, key=lambda r: (r["instance"], r["solver"])):
            sec = r.get("seconds", {}).get("solve")
            mr = asis_rel(r)
            lines.append(
                f"| {r['instance']} | {r['solver']} | {_status(r)} | "
                f"{'-' if sec is None else f'{sec:.0f}'} | "
                f"{'-' if mr is None else f'{mr:.1e}'} | "
                f"{(r.get('best') or {}).get('level', '-')} |"
            )
        lines.append("")
    return "\n".join(lines)


def _q3(vals: list[float]) -> str:
    if not vals:
        return "-"
    return f"{quantile(vals, 0.5):.1f} [{quantile(vals, 0.25):.1f}, {quantile(vals, 0.75):.1f}]"


__all__ = ["EPS", "asis_ok", "markdown", "repaired_ok", "summarise"]
