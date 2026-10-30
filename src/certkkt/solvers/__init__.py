"""Solver adapters (native APIs) returning canonical primal-dual pairs."""

from certkkt.solvers.adapters import ADAPTERS, get_adapter
from certkkt.solvers.base import STATUSES, Adapter, SolverResult

__all__ = ["ADAPTERS", "STATUSES", "Adapter", "SolverResult", "get_adapter"]
