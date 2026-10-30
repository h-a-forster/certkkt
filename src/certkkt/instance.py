"""Instances: a canonical problem plus, when it came from MPS/QPS, its boxed form."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from certkkt.problem import BoxedQP, ConicProblem, FloatData, RowOrigin


@dataclass
class Instance:
    """What solvers and the certifier work on.

    ``boxed`` and ``origins`` are set for MPS/QPS input: solvers with a native boxed form
    (HiGHS, OSQP) receive it, and their multipliers are mapped back to the conic duals of
    ``problem`` through ``origins``.
    """

    problem: ConicProblem
    boxed: BoxedQP | None = None
    origins: list[RowOrigin] | None = None
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    _float: FloatData | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.problem.name

    def float_data(self) -> FloatData:
        if self._float is None:
            self._float = self.problem.float_data()
        return self._float

    @classmethod
    def from_boxed(cls, boxed: BoxedQP, source: str = "") -> Instance:
        problem, origins = boxed.to_conic()
        return cls(problem=problem, boxed=boxed, origins=origins, source=source)


def load_instance(path: str | Path, *, exact_decimals: bool = True) -> Instance:
    """Load ``.mps``/``.qps``/``.cbf`` files, optionally gzip-compressed (``.gz``).

    With ``exact_decimals=False`` every number is first rounded to the nearest double, so
    the certified problem is exactly the one a float solver receives.
    """
    from certkkt.cbf import read_cbf
    from certkkt.qps import read_qps

    p = Path(path)
    suffixes = [s.lower() for s in p.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    ext = suffixes[-1] if suffixes else ""
    if ext in (".mps", ".qps", ".sif"):
        return Instance.from_boxed(read_qps(p, exact_decimals=exact_decimals), source=p.name)
    if ext == ".cbf":
        return Instance(problem=read_cbf(p, exact_decimals=exact_decimals), source=p.name)
    raise ValueError(f"unknown problem file type {p.name!r} (expected .mps, .qps or .cbf)")


__all__ = ["Instance", "load_instance"]
