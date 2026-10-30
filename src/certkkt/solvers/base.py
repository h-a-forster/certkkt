"""Solver adapter contract.

An adapter solves an :class:`~certkkt.instance.Instance` through the solver's native
Python API with the solver's own defaults (plus any settings passed explicitly) and
returns a :class:`SolverResult` whose ``x`` and ``z`` are in the canonical convention of
:mod:`certkkt.problem`: ``P x + q + A'z = 0`` and ``z`` in the dual cone. Adapters never
alter the solution they receive beyond this convention change (no clipping, no
projection, no rescaling).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

import numpy as np

from certkkt.instance import Instance

#: Normalised statuses. "optimal" means the solver claims convergence at its tolerances.
STATUSES = (
    "optimal",
    "inaccurate",
    "primal_infeasible",
    "dual_infeasible",
    "max_iter",
    "time_limit",
    "numerical_error",
    "error",
    "unsupported",
)


@dataclass
class SolverResult:
    solver: str
    version: str
    status: str
    raw_status: str
    x: np.ndarray | None = None
    z: np.ndarray | None = None
    objective: float | None = None
    iterations: int | None = None
    solve_time: float | None = None
    wall_time: float | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")

    @property
    def has_solution(self) -> bool:
        return self.x is not None and bool(np.isfinite(self.x).all())

    @property
    def has_dual(self) -> bool:
        return self.z is not None and bool(np.isfinite(self.z).all())

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "version": self.version,
            "status": self.status,
            "raw_status": self.raw_status,
            "objective": self.objective,
            "iterations": self.iterations,
            "solve_time": self.solve_time,
            "wall_time": self.wall_time,
            "settings": {k: _jsonable(v) for k, v in self.settings.items()},
            "message": self.message,
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, bool | int | float | str) or v is None:
        return v
    return str(v)


class Adapter(ABC):
    """Base class: subclasses set ``name``/``package`` and implement ``_solve``."""

    name: str = ""
    package: str = ""  # distribution name, for the version string
    supports_soc: bool = False
    supports_quadratic: bool = False
    #: The solver's documented default accuracy (exact decimal), in its own measure.
    default_tol: str = ""

    def version(self) -> str:
        try:
            return metadata.version(self.package)
        except metadata.PackageNotFoundError:
            return "not installed"

    def available(self) -> bool:
        try:
            self._import()
        except ImportError:
            return False
        return True

    @abstractmethod
    def _import(self) -> Any: ...

    def unsupported_reason(self, inst: Instance) -> str | None:
        p = inst.problem
        if not p.is_polyhedral and not self.supports_soc:
            return f"{self.name} does not support second-order cones"
        if p.has_quadratic_objective and not self.supports_quadratic:
            return f"{self.name} does not support a quadratic objective"
        return None

    def tolerance_settings(self, tol: float) -> dict[str, Any]:
        """Settings that request accuracy ``tol`` from this solver."""
        raise NotImplementedError

    def time_limit_settings(self, seconds: float) -> dict[str, Any]:
        """Settings that cap the solver's wall-clock time (empty if it has no such option)."""
        return {}

    def solve(self, inst: Instance, settings: dict[str, Any] | None = None) -> SolverResult:
        settings = dict(settings or {})
        reason = self.unsupported_reason(inst)
        if reason is not None:
            return SolverResult(self.name, self.version(), "unsupported", "", message=reason)
        t0 = time.perf_counter()
        try:
            res = self._solve(inst, settings)
        except Exception as exc:  # solver crashed or rejected the input
            return SolverResult(
                self.name,
                self.version(),
                "error",
                type(exc).__name__,
                settings=settings,
                wall_time=time.perf_counter() - t0,
                message=str(exc)[:500],
            )
        res.wall_time = time.perf_counter() - t0
        return res

    @abstractmethod
    def _solve(self, inst: Instance, settings: dict[str, Any]) -> SolverResult: ...

    def _result(self, status: str, raw: str, settings: dict[str, Any], **kw: Any) -> SolverResult:
        return SolverResult(self.name, self.version(), status, raw, settings=settings, **kw)


def as_array(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).ravel()
    return a
