"""Aggregate a study's units.jsonl into the tables of REPORT.md (DESIGN section 6)."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from fractions import Fraction
from typing import Any

from certkkt.stats import fmt_share, holm, mcnemar_exact, quantile, wilson

EPS = ("1e-4", "1e-6", "1e-8")
EXACT_LEVELS = ("optimal", "bounded")


def _le(v: float, eps: str) -> bool:
    """v <= eps exactly; v is a rounded-up upper bound and eps an exact decimal string."""
    return Fraction(v) <= Fraction(eps)


def default_tol(solver: str) -> str | None:
    from certkkt.solvers import ADAPTERS

    a = ADAPTERS.get(solver)
    return a.default_tol if a is not None and a.default_tol else None


def asis_rel(rec: dict[str, Any]) -> float | None:
    """Exact (rounded-up) largest relative residual of the solver's floats, or None."""
    a = rec.get("as_is")
    return None if not a else a.get("max_rel")


def asis_ok(rec: dict[str, Any], eps: str) -> bool:
    """As :meth:`certkkt.certify.Certificate.certified_at` (not on a non-PSD objective)."""
    v = asis_rel(rec)
    return v is not None and rec["as_is"]["psd"] in ("zero", "psd") and _le(v, eps)


def repaired_ok(rec: dict[str, Any], eps: str) -> bool:
    """An exactly feasible primal-dual pair whose exact relative gap is at most eps."""
    b = rec.get("best")
    if not b or b.get("level") not in EXACT_LEVELS:
        return False
    return b.get("max_rel") is not None and _le(b["max_rel"], eps)


def exactly_optimal(rec: dict[str, Any]) -> bool:
    b = rec.get("best")
    return bool(b) and b.get("level") == "optimal"


def float_max_rel(rec: dict[str, Any]) -> float | None:
    """Largest float64-evaluated relative residual, or None unless as-is and all three
    float residuals are available (the records on which the two can be compared)."""
    fl = rec.get("float") or {}
    vals = [fl.get(k) for k in ("rel_primal", "rel_dual", "rel_gap")]
    return None if asis_rel(rec) is None or None in vals else max(vals)


def float_disagrees(rec: dict[str, Any], eps: str) -> bool:
    """Float64 evaluation of the residuals and exact evaluation disagree at eps."""
    fv = float_max_rel(rec)
    return fv is not None and (fv <= float(eps)) != _le(asis_rel(rec), eps)


def _group(records: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[tuple(r.get(k) for k in keys)].append(r)
    return dict(sorted(out.items()))


def _status(rec: dict[str, Any]) -> str:
    return rec.get("result", {}).get("status") or rec.get("outcome", "?")


def _levels(levels: Any) -> str:
    counts: dict[str, int] = defaultdict(int)
    for lv in levels:
        counts[lv or "none"] += 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


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
        claims = [r for r in recs if _status(r) == "optimal" and own_rule(r) is not None]
        if claims:
            k = sum(broke_own_rule(r) for r in claims)
            row["broke_own_rule"] = {
                "k": k,
                "n": len(claims),
                "ci95": list(wilson(k, len(claims))),
                "undetermined": sum(own_rule(r)["passes"] is None for r in claims),
            }
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
    spent = defaultdict(int)
    for r in records:
        if any("time budget exhausted" in v for v in (r.get("repair_notes") or {}).values()):
            spent[r.get("suite")] += 1
    if spent:
        lines.append(
            "Repair time budget exhausted: "
            + ", ".join(f"{k} {v}" for k, v in sorted(spent.items()))
            + " units."
        )
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
            "As-is at own tol: the same test at the solver's documented default tolerance",
            "(in certkkt's measure, which is not the solver's stopping criterion).",
            "",
            "| suite | solver | n | claims optimal | as-is 1e-4 | as-is 1e-6 | as-is 1e-8 "
            "| as-is at own tol | repaired 1e-6 | repaired 1e-8 | exact optimum |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (suite, solver), recs in _group(default, "suite", "solver").items():
            n = len(recs)
            cells = [
                suite,
                solver,
                str(n),
                fmt_share(sum(_status(r) == "optimal" for r in recs), n),
                *(fmt_share(sum(asis_ok(r, e) for r in recs), n) for e in EPS),
                f"{fmt_share(sum(asis_ok(r, t) for r in recs), n)} at {t}"
                if (t := default_tol(solver))
                else "-",
                *(fmt_share(sum(repaired_ok(r, e) for r in recs), n) for e in ("1e-6", "1e-8")),
                fmt_share(sum(exactly_optimal(r) for r in recs), n),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines += _paired(default)
        comparable = [r for r in records if float_max_rel(r) is not None]
        for eps in EPS:
            dis = [r for r in comparable if float_disagrees(r, eps)]
            lines.append(
                f"Float64 and exact evaluation of the as-is residuals disagree at eps = {eps} "
                f"on {len(dis)} of {len(comparable)} units with both"
                + (": " + ", ".join(r["key"] for r in dis[:10]) if dis else "")
                + ("" if len(dis) <= 10 else ", ...")
                + "."
            )
        if comparable:
            ratio = max(
                abs(math.log10(max(float_max_rel(r), 1e-300) / max(asis_rel(r), 1e-300)))
                for r in comparable
            )
            diff = max(abs(float_max_rel(r) - asis_rel(r)) for r in comparable)
            near = ", ".join(
                f"{sum(float(e) / 2 <= asis_rel(r) <= 2 * float(e) for r in comparable)} of {e}"
                for e in EPS
            )
            lines.append(
                f"Largest |log10(float64 / exact)| of max_rel: {ratio:.2f}; largest absolute "
                f"difference: {diff:.2g}. Units with an exact max_rel within a factor of 2 "
                f"of eps: {near}."
            )
        lines.append("")
        lines.append(
            "Certificate level as returned (as-is), over units with a solution: "
            + _levels(r["as_is"].get("level") for r in records if r.get("as_is"))
            + "."
        )
        for (suite,), recs in _group(records, "suite").items():
            lines.append(
                f"Best certificate after repair, {suite}: "
                + _levels((r.get("best") or {}).get("level") for r in recs)
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

    lines += _own_rule_section(records)

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


#: The solver's float report of each own-rule criterion, where it gives the same quantity.
#: OSQP's primal residual is left out: it uses an internal iterate certkkt cannot see.
SELF_REPORT = {
    "clarabel": {"primal residual": "r_prim", "dual residual": "r_dual"},
    "scs": {"primal residual": "res_pri", "dual residual": "res_dual", "duality gap": "gap"},
    "osqp": {"dual residual": "dual_res", "duality gap": "duality_gap"},
    "highs": {
        "primal feasibility": "max_primal_infeasibility",
        "dual feasibility": "max_dual_infeasibility",
    },
}


def own_rule(rec: dict[str, Any]) -> dict[str, Any] | None:
    """The record's own-stopping-rule evaluation, or None if there is none."""
    o = rec.get("own_test")
    return o if o and "unavailable" not in o else None


def broke_own_rule(rec: dict[str, Any]) -> bool:
    """Claims optimal, yet its output fails the solver's own stopping rule exactly."""
    o = own_rule(rec)
    return _status(rec) == "optimal" and o is not None and o["passes"] is False


_VERDICT = {True: "passes", False: "fails", None: "undetermined"}


def _own_rule_section(records: list[dict[str, Any]]) -> list[str]:
    evaluated = [r for r in records if r.get("own_test")]
    if not evaluated:
        return []
    lines = [
        "## Each solver against its own stopping rule",
        "",
        "Each solver's own termination test (certkkt.own_tests: its residual definitions,",
        "norms, scaling and default tolerances, from its source) evaluated in exact",
        "arithmetic on the data it was given and the output it returned. 'Broke own rule':",
        "claims optimal, yet a condition of its own test fails exactly. 'Misses certkkt",
        "at own tol': the uniform yardstick at the value of the solver's default tolerance,",
        "same units. Largest ratio: largest value/threshold over the rule's conditions.",
        "",
        "| suite | solver | tol | claims optimal | broke own rule | undetermined "
        "| misses certkkt at own tol | largest ratio | binding condition |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tol_order = {t: i for i, t in enumerate(("default", "1e-4", "1e-6", "1e-8"))}
    groups = _group(evaluated, "suite", "solver", "tol")
    for (suite, solver, tol), recs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], tol_order.get(kv[0][2], 9))
    ):
        claims = [r for r in recs if _status(r) == "optimal" and own_rule(r) is not None]
        n = len(claims)
        if not n:
            continue
        undet = sum(own_rule(r)["passes"] is None for r in claims)
        t = default_tol(solver) if tol == "default" else tol
        miss = fmt_share(sum(not asis_ok(r, t) for r in claims), n) if t else "-"
        ratios = [max(c["ratio"] for c in own_rule(r)["criteria"]) for r in claims]
        binding: dict[str, int] = defaultdict(int)
        for r in claims:
            binding[own_rule(r)["binding"]] += 1
        lines.append(
            f"| {suite} | {solver} | {tol} | {n} | "
            f"{fmt_share(sum(broke_own_rule(r) for r in claims), n)} | {undet} | {miss} | "
            f"{max(ratios):.3f} | "
            + ", ".join(f"{k} {v}" for k, v in sorted(binding.items(), key=lambda kv: -kv[1]))
            + " |"
        )
    lines.append("")
    others = [r for r in evaluated if _status(r) != "optimal" and own_rule(r) is not None]
    if others:
        lines.append(
            "Units not claiming optimal, against the rule of the status they claim "
            "(reduced tolerances for 'inaccurate' where the solver has them): "
            + ", ".join(
                f"{r['key']} ({_status(r)}) {_VERDICT[own_rule(r)['passes']]}"
                for r in sorted(others, key=lambda r: r["key"])
            )
            + "."
        )
    undet = [r for r in evaluated if own_rule(r) is not None and own_rule(r)["passes"] is None]
    if undet:
        by_report = [
            r
            for r in undet
            if (by := own_rule(r)["diagnostics"].get("undetermined_by_solver_report"))
            and all(v["passes"] for v in by.values())
        ]
        lines.append(
            f"Undetermined on {len(undet)} units (a condition needs a quantity the solver "
            f"does not return); on {len(by_report)} of them the solver's own float report "
            "of that quantity passes the condition."
        )
    unavailable = [r for r in evaluated if own_rule(r) is None]
    if unavailable:
        lines.append(
            f"Own rule not evaluated on {len(unavailable)} units: "
            + "; ".join(f"{r['key']}: {r['own_test']['unavailable']}" for r in unavailable[:5])
            + ("" if len(unavailable) <= 5 else "; ...")
            + "."
        )
    gaps = []
    for r in evaluated:
        o = own_rule(r)
        if o is None or _status(r) != "optimal":
            continue
        keys = SELF_REPORT.get(r["solver"], {})
        for c in o["criteria"]:
            v = (o.get("solver_report") or {}).get(keys.get(c["name"], ""))
            if v is not None and c["threshold"] > 0:
                gaps.append(abs(abs(v) - c["value"]) / c["threshold"])
    if gaps:
        lines.append(
            f"On the units claiming optimal, the solvers' own float reports of the same "
            f"quantities differ from the exact values by at most {max(gaps):.1e} of the "
            f"threshold ({len(gaps)} comparisons)."
        )
    lines.append("")
    return lines


def _paired(default: list[dict[str, Any]]) -> list[str]:
    """Exact optima paired by instance: per solver pair, the problems only one of them
    reached, with an exact McNemar test and Holm adjustment within the suite."""
    lines: list[str] = []
    for (suite,), recs in _group(default, "suite").items():
        got: dict[str, dict[str, bool]] = defaultdict(dict)
        for r in recs:
            got[r["solver"]][r["instance"]] = exactly_optimal(r)
        solvers = sorted(got)
        common = sorted(set.intersection(*(set(v) for v in got.values())))
        union = sum(any(got[s][i] for s in solvers) for i in common)
        if len(solvers) < 2 or union == 0:
            continue
        inter = sum(all(got[s][i] for s in solvers) for i in common)
        rows = []
        for a, b in itertools.combinations(solvers, 2):
            oa = sum(got[a][i] and not got[b][i] for i in common)
            ob = sum(got[b][i] and not got[a][i] for i in common)
            rows.append((a, b, oa, ob, mcnemar_exact(oa, ob)))
        adj = holm([row[4] for row in rows])
        lines += [
            f"Exact optimum paired by instance ({suite}, {len(common)} problems): "
            f"{union} reached from at least one solver, {inter} from all "
            f"{len(solvers)}. Exact McNemar test, Holm-adjusted over the "
            f"{len(rows)} pairs.",
            "",
            "| a | b | only a | only b | p | Holm p |",
            "|---|---|---|---|---|---|",
            *(
                f"| {a} | {b} | {oa} | {ob} | {p:.3f} | {h:.3f} |"
                for (a, b, oa, ob, p), h in zip(rows, adj, strict=True)
            ),
            "",
        ]
    return lines


def _q3(vals: list[float]) -> str:
    if not vals:
        return "-"
    return f"{quantile(vals, 0.5):.1f} [{quantile(vals, 0.25):.1f}, {quantile(vals, 0.75):.1f}]"


__all__ = ["EPS", "asis_ok", "broke_own_rule", "markdown", "repaired_ok", "summarise"]
