"""Conic data from a cvxpy problem.

cvxpy compiles a problem into the canonical form of :mod:`certkkt.problem`
(``A x + s = b``, ``s`` in zero x nonneg x SOC cones, quadratic objective ``1/2 x'Px +
q'x``) for solvers such as Clarabel. :func:`from_cvxpy` takes that compiled data as the
exact problem (the floats cvxpy produces convert to rationals exactly), so certificates
refer to the problem the solver actually receives. Variables are cvxpy's internal ones,
which may differ from the user's (cvxpy adds epigraph variables).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from certkkt.instance import Instance
from certkkt.problem import Cone, make_problem


def from_cvxpy(problem: Any, name: str = "cvxpy") -> Instance:
    """Instance from a ``cvxpy.Problem`` (compiled through cvxpy's Clarabel path).

    Raises ValueError for cones certkkt does not certify (PSD, exponential, power).
    """
    import cvxpy as cp

    data, _chain, _inv = problem.get_problem_data(cp.CLARABEL)
    dims = data["dims"]
    if dims.psd or dims.exp or getattr(dims, "p3d", None):
        raise ValueError("only zero, nonnegative and second-order cones are supported")
    cones: list[Cone] = []
    if dims.zero:
        cones.append(Cone("zero", int(dims.zero)))
    if dims.nonneg:
        cones.append(Cone("nonneg", int(dims.nonneg)))
    for k in dims.soc:
        cones.append(Cone("soc", int(k)) if int(k) > 1 else Cone("nonneg", 1))
    A = sp.csr_array(data["A"])
    n = A.shape[1]
    P = data.get("P")
    P = sp.csr_array((n, n)) if P is None else sp.csr_array(P)
    if (abs(P - P.T)).nnz:  # cvxpy may hand over one triangle: symmetrise it
        P = sp.csr_array(sp.triu(P) + sp.triu(P, 1).T)
    offset = _offset(data)
    prob = make_problem(
        A=A,
        b=np.asarray(data["b"], dtype=np.float64),
        cones=cones,
        q=np.asarray(data["c"], dtype=np.float64),
        P=P,
        r=offset,
        name=name,
    )
    return Instance(problem=prob, source="cvxpy", meta={"cvxpy_dims": str(dims)})


def _offset(data: dict[str, Any]) -> float:
    pp = data.get("param_prob")
    if pp is None:
        return 0.0
    out = pp.apply_parameters(quad_obj="P" in data and data["P"] is not None)
    d = out[2] if len(out) == 5 else out[1]
    return float(np.asarray(d))


__all__ = ["from_cvxpy"]
