"""Command-line interface: ``certkkt solve | certify | solvers | fetch | study``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _cert_lines(cert: Any) -> list[str]:
    d = cert.to_dict()

    def g(v: Any) -> str:
        return "-" if v is None else f"{v:.3e}"

    return [
        f"  [{d['method']}] level={d['level']}  psd={d['psd']}",
        f"    primal: eq={g(d['primal_eq_residual'])} cone={g(d['primal_cone_violation'])}"
        f" rel={g(d['rel_primal'])}",
        f"    dual:   stat={g(d['stationarity_residual'])} cone={g(d['dual_cone_violation'])}"
        f" rel={g(d['rel_dual'])}",
        f"    pobj={d['pobj']!r} dobj={d['dobj']!r} gap<={g(d['gap_up'])}"
        f" rel_gap={g(d['rel_gap'])} max_rel={g(d['max_rel'])}",
        *(f"    note: {n}" for n in d["notes"]),
    ]


def _report(inst: Any, x: np.ndarray, z: np.ndarray | None, args: argparse.Namespace) -> int:
    from certkkt.certify import certify, meets_level
    from certkkt.repair import repair

    cert = certify(inst.problem, x, z)
    best = None
    if not args.no_repair:
        best = repair(inst.problem, x, z, time_budget=args.time_budget).best
    final = best if best is not None and best.rank() < cert.rank() else cert
    if args.json:
        out = {"problem": inst.problem.summary(), "as_is": cert.to_dict()}
        if best is not None:
            out["best"] = best.to_dict()
        print(json.dumps(out, indent=2))
    else:
        print(f"problem {inst.name}: {inst.problem.summary()}")
        print("as is:")
        print("\n".join(_cert_lines(cert)))
        if best is not None:
            print("best after repair:")
            print("\n".join(_cert_lines(best)))
    failed = False
    if args.require_eps is not None and not final.certified_at(args.require_eps):
        print(f"FAIL: not certified at eps={args.require_eps}", file=sys.stderr)
        failed = True
    if args.require_level is not None and not any(
        meets_level(c.level, args.require_level) for c in (cert, best) if c is not None
    ):
        print(f"FAIL: level {final.level} does not meet {args.require_level}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


def cmd_solve(args: argparse.Namespace) -> int:
    from certkkt.instance import load_instance
    from certkkt.solvers import get_adapter

    adapter = get_adapter(args.solver)
    if not adapter.available():
        print(
            f"{args.solver} is not installed (pip install certkkt[{args.solver}])", file=sys.stderr
        )
        return 2
    inst = load_instance(args.problem)
    settings: dict[str, Any] = {}
    if args.tol is not None:
        settings.update(adapter.tolerance_settings(args.tol))
    for kv in args.set or []:
        k, sep, v = kv.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        settings[k] = _literal(v)
    res = adapter.solve(inst, settings)
    took = "" if res.wall_time is None else f" time={res.wall_time:.2f}s"
    print(
        f"{res.solver} {res.version}: {res.status} ({res.raw_status}) "
        f"objective={res.objective}{took}",
        file=sys.stderr,
    )
    if not res.has_solution:
        print(f"no solution to certify: {res.message}", file=sys.stderr)
        return 1
    if args.save_solution:
        Path(args.save_solution).write_text(
            json.dumps({"x": res.x.tolist(), "z": None if res.z is None else res.z.tolist()})
        )
    return _report(inst, res.x, res.z if res.has_dual else None, args)


def _literal(v: str) -> Any:
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def load_solution(path: Path, inst: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Read ``{"x": [...], "z": [...]}`` (canonical conic duals) or, for MPS/QPS input,
    ``{"x": [...], "y_row": [...], "y_col": [...]}`` (boxed multipliers, OSQP sign
    convention: positive where an upper bound is active)."""
    from certkkt.problem import boxed_duals_to_conic

    data = json.loads(path.read_text())
    if "x" not in data:
        raise ValueError("solution file needs an 'x' array")
    x = np.asarray(data["x"], dtype=np.float64)
    if "z" in data and ("y_row" in data or "y_col" in data):
        raise ValueError("give either 'z' or 'y_row'/'y_col', not both")
    if "y_row" in data or "y_col" in data:
        if inst.origins is None or inst.boxed is None:
            raise ValueError("'y_row'/'y_col' need an MPS/QPS problem")
        y_row = np.asarray(data.get("y_row") or np.zeros(inst.boxed.m), dtype=np.float64)
        y_col = np.asarray(data.get("y_col") or np.zeros(inst.boxed.n), dtype=np.float64)
        if y_row.shape != (inst.boxed.m,) or y_col.shape != (inst.boxed.n,):
            raise ValueError("y_row/y_col have the wrong length")
        return x, boxed_duals_to_conic(inst.origins, y_row, y_col)
    z = data.get("z")
    return x, None if z is None else np.asarray(z, dtype=np.float64)


def cmd_certify(args: argparse.Namespace) -> int:
    from certkkt.instance import load_instance

    inst = load_instance(args.problem)
    x, z = load_solution(Path(args.solution), inst)
    return _report(inst, x, z, args)


def cmd_solvers(args: argparse.Namespace) -> int:
    from certkkt.rational import BACKEND
    from certkkt.solvers import ADAPTERS

    print(f"rational backend: {BACKEND}")
    for name, a in ADAPTERS.items():
        kinds = (
            ["LP"] + (["QP"] if a.supports_quadratic else []) + (["SOCP"] if a.supports_soc else [])
        )
        print(f"{name:9s} {a.version():14s} {'/'.join(kinds)}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from certkkt.datasets import fetch

    names = set(args.only) if args.only else None
    for suite in args.suites:
        paths = fetch(suite, Path(args.data), names=names)
        print(f"{suite}: {len(paths)} files ok")
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    from certkkt import study

    run_dir = Path(args.run_dir or f"runs/{args.profile}")
    if args.action == "run":
        suites = tuple(args.suite) if args.suite else ("mm", "cblib", "portfolio")
        study.run_study(
            args.profile,
            suites,
            run_dir=run_dir,
            data_root=Path(args.data),
            workers=args.workers,
            max_units=args.max_units,
        )
        return 0
    path = run_dir / "units.jsonl"
    if not path.exists():
        print(f"no records at {path}", file=sys.stderr)
        return 1
    records = study.read_records(path)
    if args.action == "status":
        suites = tuple(args.suite) if args.suite else study.SUITES
        planned = study.plan(study.PROFILES[args.profile], suites, Path(args.data))
        done = {r["key"] for r in records}
        print(f"{sum(u.key in done for u in planned)}/{len(planned)} units done")
        return 0
    from certkkt.report import markdown, summarise

    text = markdown(records, title=f"certkkt study: {args.profile} profile")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(summarise(records), indent=1) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from certkkt import __version__
    from certkkt.certify import LEVELS

    p = argparse.ArgumentParser(
        prog="certkkt", description="Exact optimality certificates for LP/QP/SOCP solutions."
    )
    p.add_argument("--version", action="version", version=f"certkkt {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_report_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--no-repair", action="store_true", help="certify the floats only")
        sp.add_argument(
            "--time-budget",
            type=float,
            default=60.0,
            help="seconds for all repair methods together (default 60)",
        )
        sp.add_argument("--json", action="store_true", help="print JSON")
        sp.add_argument("--require-eps", help="exit 1 unless certified at this eps")
        sp.add_argument(
            "--require-level",
            choices=LEVELS,
            help="exit 1 unless the best certificate is at least this strong",
        )

    sp = sub.add_parser("solve", help="solve a problem file and certify the result")
    sp.add_argument("problem", help=".mps/.qps/.cbf file (optionally .gz)")
    sp.add_argument("--solver", required=True)
    sp.add_argument("--tol", type=float, help="requested accuracy (solver-specific mapping)")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="solver setting")
    sp.add_argument("--save-solution", help="write x and z (canonical convention) as JSON")
    add_report_opts(sp)
    sp.set_defaults(func=cmd_solve)

    sp = sub.add_parser("certify", help="certify a given solution of a problem file")
    sp.add_argument("problem")
    sp.add_argument("solution", help="JSON with x and z (or y_row/y_col for MPS/QPS)")
    add_report_opts(sp)
    sp.set_defaults(func=cmd_certify)

    sp = sub.add_parser("solvers", help="list solver adapters and installed versions")
    sp.set_defaults(func=cmd_solvers)

    sp = sub.add_parser("fetch", help="download benchmark data (checksummed)")
    sp.add_argument("suites", nargs="+", choices=("mm", "cblib", "mittelmann", "french"))
    sp.add_argument("--only", nargs="*", help="restrict to these file names")
    sp.add_argument("--data", default="data")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("study", help="run, inspect or report the certification study")
    sp.add_argument("action", choices=("run", "status", "report"))
    sp.add_argument("--profile", default="quick", choices=("quick", "full"))
    sp.add_argument("--suite", action="append", choices=("mm", "cblib", "portfolio", "mittelmann"))
    sp.add_argument("--run-dir", help="default runs/<profile>")
    sp.add_argument("--data", default="data")
    sp.add_argument("--workers", type=int, default=2)
    sp.add_argument("--max-units", type=int, help="stop after this many new units")
    sp.add_argument("--out", help="report: write markdown here")
    sp.add_argument("--json", help="report: write the summary JSON here")
    sp.set_defaults(func=cmd_study)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
