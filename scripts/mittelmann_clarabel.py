"""Clarabel on the instances of Mittelmann's Large SOCP Benchmark.

A check of the "Clarabel solves 14 of 18" figure (Li, Huang, Liu, Ge, Ye, arXiv
2608.09159, Table 7: tolerances 1e-4 and 1e-6, 3600 s limit; REPORT.md section 5). The
instances are read straight into doubles (``cbf.read_cbf_float``), so even the large ones
load; nothing here is certified exactly except, up to ``--own-test-max-nnz`` nonzeros,
Clarabel's own stopping rule (``own_tests``). Each (instance, tolerance) unit appends one
JSON line to ``<out>/units.jsonl``; finished units are skipped on rerun; no new unit
starts once ``--budget-seconds`` have passed. Progress goes to ``<out>/progress.log``.

    uv run certkkt fetch mittelmann --only 2013_firL2a
    uv run python scripts/mittelmann_clarabel.py --out runs/mittelmann-clarabel \
        --instance chainsing-50000-3 --tol 1e-6 --tol 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from certkkt.cbf import read_cbf_float
    from certkkt.solvers import get_adapter
    from certkkt.study import _finite, own_test_record, versions

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--instance", action="append", required=True, help="CBLIB file stem")
    ap.add_argument("--tol", action="append", required=True, help="1e-4, 1e-6 or default")
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--budget-seconds", type=float, default=float("inf"))
    ap.add_argument("--own-test-max-nnz", type=int, default=2_000_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    units = args.out / "units.jsonl"
    done = set()
    if units.exists():
        done = {json.loads(line)["key"] for line in units.read_text().splitlines() if line}
    clarabel = get_adapter("clarabel")
    started = time.perf_counter()

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with (args.out / "progress.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"clarabel {clarabel.version()} on {platform.platform(terse=True)}; {versions()}")
    for name in args.instance:
        todo = [t for t in args.tol if f"{name}/{t}" not in done]
        if not todo:
            continue
        if time.perf_counter() - started > args.budget_seconds:
            log(f"budget spent; not starting {name}")
            break
        t0 = time.perf_counter()
        fd, cones = read_cbf_float(args.data / "raw" / "cblib" / f"{name}.cbf.gz")
        load_s = time.perf_counter() - t0
        nnz = int(fd.A.nnz)
        log(f"{name}: read in {load_s:.0f} s, nnz {nnz}")
        # What ClarabelAdapter reads from an Instance: the float data and the cones.
        inst = SimpleNamespace(
            float_data=lambda fd=fd: fd,
            problem=SimpleNamespace(
                cones=cones, is_polyhedral=False, has_quadratic_objective=False
            ),
        )
        for tol in todo:
            if time.perf_counter() - started > args.budget_seconds:
                log(f"budget spent; not starting {name}/{tol}")
                break
            settings = {} if tol == "default" else clarabel.tolerance_settings(float(tol))
            settings.update(clarabel.time_limit_settings(args.time_limit))
            res = clarabel.solve(inst, settings)
            rec = {
                "key": f"{name}/{tol}",
                "instance": name,
                "tol": tol,
                "nnz": nnz,
                "result": res.to_dict(),
                "read_seconds": round(load_s, 1),
            }
            if res.has_solution and nnz <= args.own_test_max_nnz:
                rec["own_test"] = own_test_record(res)
            res.native = None  # free the solver's copy of the data
            with units.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_finite(rec), default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            log(f"{name}/{tol}: {res.raw_status} in {res.solve_time} s, {res.iterations} iters")
        del fd, inst
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
