"""Checkpointed certification study (DESIGN section 5).

A study is a list of units ``(suite, instance, solver, tolerance)``. Each unit loads the
instance, runs the solver through its native API, certifies the returned floats exactly,
runs the repairs, and appends one JSON record to ``runs/<study>/units.jsonl``. Completed
units are skipped on rerun, so an interrupted study resumes where it stopped. Units run
in persistent worker processes with a hard per-unit timeout: a hung solver or an exact
computation that outgrows its budget is killed and recorded as a timeout, and the worker
is replaced.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import queue
import sys
import threading
import time
import traceback
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib import metadata
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

SUITES = ("mm", "cblib", "portfolio", "mittelmann")
SUITE_SOLVERS = {
    "mm": ("clarabel", "osqp", "scs", "highs"),
    "cblib": ("clarabel", "scs", "ecos"),
    "portfolio": ("clarabel", "scs", "ecos"),
    "mittelmann": ("clarabel", "scs", "ecos"),
}
TOLERANCES = ("default", "1e-4", "1e-6", "1e-8")


@dataclass(frozen=True)
class Limits:
    """Per-unit limits: solver wall-clock limit (where the solver has one), total repair
    time, and the hard timeout after which the worker is killed. ``certify_max_nnz``
    skips exact work on larger instances (status and float residuals only), and
    ``max_file_bytes`` skips instances whose file is too large to parse into exact
    rationals in memory."""

    solver_seconds: float = 60.0
    repair_seconds: float = 30.0
    hard_seconds: float = 300.0
    certify_max_nnz: int = 2_000_000
    #: skip (without loading) instance files larger than this (compressed bytes)
    max_file_bytes: int | None = None
    #: run the repairs on tolerance-sweep units too (not only on tol = "default")
    repair_sweep: bool = True


@dataclass(frozen=True)
class Profile:
    name: str
    #: suite -> tolerances run on that suite
    tolerances: dict[str, tuple[str, ...]]
    #: Maros-Meszaros: largest uncompressed QPS file size included (bytes)
    mm_max_bytes: int | None
    cblib: tuple[str, ...]
    portfolio: tuple[str, ...] | None  # None = whole family
    mittelmann: tuple[str, ...]
    limits: dict[str, Limits] = field(default_factory=dict)

    def limits_for(self, suite: str) -> Limits:
        return self.limits.get(suite, Limits())


PROFILES = {
    "quick": Profile(
        name="quick",
        tolerances={
            "mm": ("default",),
            "cblib": ("default",),
            "portfolio": TOLERANCES,
            "mittelmann": (),
        },
        mm_max_bytes=100_000,
        cblib=("nql30", "qssp30", "sched_50_50_orig", "sched_50_50_scaled"),
        portfolio=tuple(
            f"pf10_{end}_L{lev}_T0.25" for end in (200512, 201512, 202512) for lev in ("1.3", "2")
        ),
        mittelmann=(),
        limits={s: Limits(60.0, 30.0, 300.0) for s in ("mm", "cblib", "portfolio")},
    ),
    "full": Profile(
        name="full",
        tolerances={
            "mm": TOLERANCES,
            "cblib": TOLERANCES,
            "portfolio": TOLERANCES,
            "mittelmann": ("default",),
        },
        mm_max_bytes=None,
        cblib=tuple(),  # filled below: the whole DIMACS subset
        portfolio=None,
        mittelmann=tuple(),  # filled below: every downloadable instance
        limits={
            "mm": Limits(600.0, 60.0, 1200.0, repair_sweep=False),
            "cblib": Limits(600.0, 60.0, 1200.0, repair_sweep=False),
            "portfolio": Limits(600.0, 60.0, 1200.0),
            "mittelmann": Limits(
                3600.0, 300.0, 7200.0, certify_max_nnz=8_000_000, max_file_bytes=30_000_000
            ),
        },
    ),
}


def _complete_full() -> None:
    from certkkt.datasets import CBLIB_DIMACS, CBLIB_MITTELMANN

    full = PROFILES["full"]
    PROFILES["full"] = Profile(
        name=full.name,
        tolerances=full.tolerances,
        mm_max_bytes=None,
        cblib=tuple(CBLIB_DIMACS),
        portfolio=None,
        mittelmann=tuple(CBLIB_MITTELMANN),
        limits=full.limits,
    )


_complete_full()


@dataclass(frozen=True)
class Unit:
    suite: str
    instance: str
    solver: str
    tol: str

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.instance}/{self.solver}/{self.tol}"


def instance_names(profile: Profile, suite: str, data_root: Path) -> list[str]:
    from certkkt.datasets import mm_members
    from certkkt.portfolio import FAMILY

    if suite == "mm":
        out = [
            name
            for name, _, _, size in mm_members(data_root)
            if profile.mm_max_bytes is None or size <= profile.mm_max_bytes
        ]
        return sorted(out)
    if suite == "cblib":
        return list(profile.cblib)
    if suite == "mittelmann":
        return list(profile.mittelmann)
    if suite == "portfolio":
        names = [s.name for s in FAMILY]
        if profile.portfolio is None:
            return names
        unknown = set(profile.portfolio) - set(names)
        if unknown:
            raise ValueError(f"portfolio instances not in the family: {sorted(unknown)}")
        return list(profile.portfolio)
    raise ValueError(f"unknown suite {suite!r}")


def plan(profile: Profile, suites: tuple[str, ...], data_root: Path) -> list[Unit]:
    units = []
    for suite in suites:
        if suite not in SUITES:
            raise ValueError(f"unknown suite {suite!r}; choose from {SUITES}")
        tols = profile.tolerances.get(suite, ())
        for inst in instance_names(profile, suite, data_root):
            for solver in SUITE_SOLVERS[suite]:
                for tol in tols:
                    units.append(Unit(suite, inst, solver, tol))
    return units


# -- loading -----------------------------------------------------------------------------


def load_unit_instance(suite: str, name: str, data_root: Path) -> Any:
    from certkkt.instance import Instance, load_instance

    if suite == "mm":
        from certkkt.datasets import mm_members
        from certkkt.qps import parse_qps

        for nm, arch, member, _ in mm_members(data_root):
            if nm == name:
                with zipfile.ZipFile(arch) as zf:
                    text = zf.read(member).decode("latin-1")
                return Instance.from_boxed(parse_qps(text, default_name=name), source=member)
        raise FileNotFoundError(f"{name} not in the Maros-Meszaros archives")
    if suite in ("cblib", "mittelmann"):
        return load_instance(data_root / "raw" / "cblib" / f"{name}.cbf.gz")
    if suite == "portfolio":
        from certkkt.portfolio import FAMILY, build_instance

        spec = next((s for s in FAMILY if s.name == name), None)
        if spec is None:
            raise ValueError(f"unknown portfolio instance {name!r}")
        return build_instance(spec, data_root / "raw" / "french")
    raise ValueError(f"unknown suite {suite!r}")


# -- one unit ----------------------------------------------------------------------------


def versions() -> dict[str, str]:
    from certkkt.rational import BACKEND

    out = {"python": platform.python_version(), "rational_backend": BACKEND}
    for dist in ("certkkt", "numpy", "scipy", "gmpy2"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = "not installed"
    return out


def run_unit(unit: Unit, inst: Any, limits: Limits, timings: dict[str, float]) -> dict[str, Any]:
    """Solve, certify and repair one unit; returns the JSON record (without timeout info)."""
    import numpy as np

    from certkkt.certify import certify, float_residuals
    from certkkt.repair import repair
    from certkkt.solvers import get_adapter

    adapter = get_adapter(unit.solver)
    settings: dict[str, Any] = {}
    if unit.tol != "default":
        settings.update(adapter.tolerance_settings(float(unit.tol)))
    settings.update(adapter.time_limit_settings(limits.solver_seconds))
    prob = inst.problem
    rec: dict[str, Any] = {
        "key": unit.key,
        **asdict(unit),
        "summary": prob.summary(),
        "limits": asdict(limits),
    }
    t0 = time.perf_counter()
    res = adapter.solve(inst, settings)
    timings["solve"] = time.perf_counter() - t0
    rec["result"] = res.to_dict()
    if not res.has_solution:
        rec["outcome"] = "no_solution"
        return rec
    x = res.x
    z = res.z if res.has_dual else None
    rec["float"] = float_residuals(prob, x, z)
    if prob.A.nnz + prob.P.nnz > limits.certify_max_nnz:
        rec["outcome"] = "too_large_for_exact"
        return rec
    t0 = time.perf_counter()
    cert = certify(prob, x, z)
    timings["certify"] = time.perf_counter() - t0
    rec["as_is"] = cert.to_dict()
    if unit.tol != "default" and not limits.repair_sweep:
        rec["outcome"] = "certified_as_is"
        return rec
    t0 = time.perf_counter()
    out = repair(prob, np.asarray(x), z, time_budget=limits.repair_seconds)
    timings["repair"] = time.perf_counter() - t0
    rec["best"] = out.best.to_dict()
    rec["methods"] = {
        c.method: {"level": c.level, "max_rel": c.to_dict()["max_rel"]} for c in out.certificates
    }
    rec["repair_notes"] = out.notes
    rec["repair_seconds"] = {k: round(v, 3) for k, v in out.seconds.items()}
    rec["outcome"] = "certified"
    return rec


def _worker(conn: Connection, data_root: str) -> None:
    """Worker loop: receive (instance, [unit dicts], limits dict), send one record each."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    root = Path(data_root)
    while True:
        msg = conn.recv()
        if msg is None:
            return
        suite, name, unit_dicts, lim = msg
        limits = Limits(**lim)
        t0 = time.perf_counter()
        inst = None
        load_error = None
        outcome = "load_error"
        path = root / "raw" / "cblib" / f"{name}.cbf.gz"
        if (
            limits.max_file_bytes is not None
            and suite in ("cblib", "mittelmann")
            and path.exists()
            and path.stat().st_size > limits.max_file_bytes
        ):
            outcome = "skipped_too_large"
            load_error = f"{path.stat().st_size} bytes > {limits.max_file_bytes}"
        else:
            try:
                inst = load_unit_instance(suite, name, root)
            except Exception as exc:  # malformed or unsupported instance: record, go on
                load_error = f"{type(exc).__name__}: {exc}"[:500]
        load_s = time.perf_counter() - t0
        for ud in unit_dicts:
            unit = Unit(**ud)
            timings: dict[str, float] = {"load": load_s}
            if inst is None:
                rec = {"key": unit.key, **ud, "outcome": outcome, "error": load_error}
            else:
                try:
                    rec = run_unit(unit, inst, limits, timings)
                except Exception as exc:
                    rec = {
                        "key": unit.key,
                        **ud,
                        "outcome": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                        "traceback": traceback.format_exc()[-2000:],
                    }
            rec["seconds"] = {k: round(v, 3) for k, v in timings.items()}
            conn.send(rec)
            load_s = 0.0


# -- driver --------------------------------------------------------------------------------


def completed_keys(path: Path) -> set[str]:
    """Keys of the records already in a units.jsonl (a torn last line is ignored)."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def read_records(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _finite(v: Any) -> Any:
    """Copy of a record with non-finite floats (a solver's nan or inf objective, float64
    residuals that overflowed) replaced by None, which strict JSON can hold."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_finite(x) for x in v]
    return v


class _Sink:
    """Appends records and progress lines, thread-safe, flushed to disk per record."""

    def __init__(self, run_dir: Path, total: int, already: int, echo: Callable[[str], None]):
        self.units = run_dir / "units.jsonl"
        self.log_path = run_dir / "progress.log"
        self.total = total
        self.done = already
        self.started = time.time()
        self.new = 0
        self.lock = threading.Lock()
        self.echo = echo
        self._repair_torn_tail()

    def _repair_torn_tail(self) -> None:
        if self.units.exists() and self.units.stat().st_size:
            data = self.units.read_bytes()
            if not data.endswith(b"\n"):
                keep = data[: data.rfind(b"\n") + 1]
                self.units.write_bytes(keep)

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self.lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self.echo(line)

    def record(self, rec: dict[str, Any]) -> None:
        line = json.dumps(_finite(rec), separators=(",", ":"), allow_nan=False, default=str)
        with self.lock:
            with self.units.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.done += 1
            self.new += 1
            el = time.time() - self.started
            rate = self.new / el if el > 0 else 0.0
            eta = (self.total - self.done) / rate if rate > 0 else float("inf")
            done, total = self.done, self.total
        best = rec.get("best", {}) or {}
        self.log(
            f"[{done}/{total}] {rec['key']} {rec.get('outcome')} "
            f"{rec.get('result', {}).get('status', '')} "
            f"best={best.get('level', '-')} rate={rate * 60:.1f}/min eta={eta / 60:.1f}min"
        )


def run_study(
    profile_name: str,
    suites: tuple[str, ...],
    *,
    run_dir: Path,
    data_root: Path,
    workers: int = 2,
    max_units: int | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    """Run (or resume) a study; returns the number of units completed in this call."""
    if profile_name not in PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}; choose from {sorted(PROFILES)}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    profile = PROFILES[profile_name]
    units = plan(profile, suites, data_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    done = completed_keys(run_dir / "units.jsonl")
    remaining = [u for u in units if u.key not in done]
    todo = remaining if max_units is None else remaining[:max_units]
    sink = _Sink(run_dir, len(units), len(units) - len(remaining), echo)
    # One manifest entry per call: versions can differ between resumed calls.
    manifest = run_dir / "manifest.json"
    calls = json.loads(manifest.read_text())["calls"] if manifest.exists() else []
    calls.append(
        {
            "profile": profile_name,
            "suites": list(suites),
            "versions": versions(),
            "solvers": _solver_versions(),
            "platform": platform.platform(terse=True),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    manifest.write_text(json.dumps({"calls": calls}, indent=2))
    sink.log(
        f"profile={profile_name} suites={','.join(suites)} units={len(units)} "
        f"done={len(units) - len(remaining)} todo={len(todo)} workers={workers}"
    )
    # Group by instance so each worker loads an instance once.
    groups: dict[tuple[str, str], list[Unit]] = {}
    for u in todo:
        groups.setdefault((u.suite, u.instance), []).append(u)
    work: queue.Queue[tuple[str, str, list[Unit]]] = queue.Queue()
    for (suite, name), us in groups.items():
        work.put((suite, name, us))
    errors: list[BaseException] = []

    def drive() -> None:
        try:
            _drive(work, profile, data_root, sink)
        except BaseException as exc:  # re-raised below so the run does not report success
            errors.append(exc)

    threads = [
        threading.Thread(target=drive, daemon=True)
        for _ in range(min(workers, max(1, len(groups))))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        sink.log(f"failed: {errors[0]!r}; {sink.new} new units recorded")
        raise RuntimeError("a study driver thread failed") from errors[0]
    sink.log(f"finished: {sink.new} new units, {sink.done}/{sink.total} total")
    return sink.new


def _solver_versions() -> dict[str, str]:
    from certkkt.solvers import ADAPTERS

    return {name: a.version() for name, a in ADAPTERS.items()}


def _spawn(data_root: Path) -> tuple[Any, Connection]:
    ctx = get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_worker, args=(child, str(data_root)), daemon=True)
    proc.start()
    child.close()
    return proc, parent


def _drive(work: queue.Queue, profile: Profile, data_root: Path, sink: _Sink) -> None:
    proc, conn = _spawn(data_root)
    try:
        while True:
            try:
                suite, name, us = work.get_nowait()
            except queue.Empty:
                return
            limits = profile.limits_for(suite)
            pending = list(us)
            while pending:
                conn.send((suite, name, [asdict(u) for u in pending], asdict(limits)))
                while pending:
                    if conn.poll(limits.hard_seconds):
                        try:
                            rec = conn.recv()
                        except EOFError:
                            rec = None
                        if rec is not None:
                            pending = [u for u in pending if u.key != rec["key"]]
                            sink.record(rec)
                            continue
                        outcome, msg = "crash", "worker exited"
                    else:
                        outcome, msg = "timeout", f"exceeded {limits.hard_seconds:.0f}s"
                    unit = pending.pop(0)
                    sink.record(
                        {
                            "key": unit.key,
                            **asdict(unit),
                            "outcome": outcome,
                            "error": msg,
                            "limits": asdict(limits),
                        }
                    )
                    proc.kill()
                    proc.join()
                    proc, conn = _spawn(data_root)
                    break  # resend the remaining units to the fresh worker
    finally:
        with contextlib.suppress(BrokenPipeError, OSError):
            conn.send(None)
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()


__all__ = [
    "Limits",
    "PROFILES",
    "Profile",
    "SUITES",
    "SUITE_SOLVERS",
    "TOLERANCES",
    "Unit",
    "completed_keys",
    "load_unit_instance",
    "plan",
    "read_records",
    "run_study",
    "run_unit",
    "versions",
]

if __name__ == "__main__":  # pragma: no cover
    sys.exit("use the certkkt CLI: certkkt study run")
