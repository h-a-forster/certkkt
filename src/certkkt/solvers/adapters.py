"""Adapters for Clarabel, SCS, OSQP, ECOS and HiGHS (native Python APIs)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from certkkt.instance import Instance
from certkkt.problem import boxed_duals_to_conic, upper_triangle
from certkkt.solvers.base import Adapter, NativeRun, SolverResult, as_array


def _cone_rows(inst: Instance) -> dict[str, list[int]]:
    """Row indices of each cone kind, SOC blocks kept contiguous and in order."""
    out: dict[str, list[int]] = {"zero": [], "nonneg": [], "soc": []}
    for cone, a, b in inst.problem.cone_slices():
        out[cone.kind].extend(range(a, b))
    return out


def _csc32(mat: Any) -> sp.csc_matrix:
    """CSC matrix with 32-bit indices (OSQP's C interface rejects 64-bit ones)."""
    out = sp.csc_matrix(mat)
    out.indices = out.indices.astype(np.int32)
    out.indptr = out.indptr.astype(np.int32)
    return out


def _soc_dims(inst: Instance) -> list[int]:
    return [c.dim for c in inst.problem.cones if c.kind == "soc"]


class ClarabelAdapter(Adapter):
    name = "clarabel"
    package = "clarabel"
    default_tol = "1e-8"
    supports_soc = True
    supports_quadratic = True

    def _import(self) -> Any:
        import clarabel

        return clarabel

    def time_limit_settings(self, seconds: float) -> dict[str, Any]:
        return {"time_limit": float(seconds)}

    def tolerance_settings(self, tol: float) -> dict[str, Any]:
        return {"tol_gap_abs": tol, "tol_gap_rel": tol, "tol_feas": tol}

    def _solve(self, inst: Instance, settings: dict[str, Any]) -> SolverResult:
        clarabel = self._import()
        fd = inst.float_data()
        cones = []
        for c in inst.problem.cones:
            ctor = {
                "zero": clarabel.ZeroConeT,
                "nonneg": clarabel.NonnegativeConeT,
                "soc": clarabel.SecondOrderConeT,
            }[c.kind]
            cones.append(ctor(c.dim))
        st = clarabel.DefaultSettings()
        st.verbose = False
        for k, v in settings.items():
            if not hasattr(st, k):
                raise ValueError(f"unknown Clarabel setting {k!r}")
            setattr(st, k, v)
        P, A = upper_triangle(fd.P), sp.csc_matrix(fd.A)
        solver = clarabel.DefaultSolver(P, fd.q, A, fd.b, cones, st)
        sol = solver.solve()
        native = NativeRun(
            form="conic",
            data={
                "P": P,
                "P_triangle": "upper",
                "q": fd.q,
                "A": A,
                "b": fd.b,
                "cones": [(c.kind, c.dim) for c in inst.problem.cones],
            },
            output={k: as_array(getattr(sol, k)) for k in ("x", "z", "s")},
            settings={
                k: getattr(st, k)
                for k in (
                    "tol_gap_abs",
                    "tol_gap_rel",
                    "tol_feas",
                    "reduced_tol_gap_abs",
                    "reduced_tol_gap_rel",
                    "reduced_tol_feas",
                    "presolve_enable",
                )
            }
            | {"infinity": float(clarabel.get_infinity())},
            info={"r_prim": float(sol.r_prim), "r_dual": float(sol.r_dual)},
        )
        raw = str(sol.status)
        status = {
            "Solved": "optimal",
            "AlmostSolved": "inaccurate",
            "PrimalInfeasible": "primal_infeasible",
            "AlmostPrimalInfeasible": "primal_infeasible",
            "DualInfeasible": "dual_infeasible",
            "AlmostDualInfeasible": "dual_infeasible",
            "MaxIterations": "max_iter",
            "MaxTime": "time_limit",
            "NumericalError": "numerical_error",
            "InsufficientProgress": "numerical_error",
        }.get(raw, "error")
        return self._result(
            status,
            raw,
            settings,
            x=as_array(sol.x),
            z=as_array(sol.z),
            objective=float(sol.obj_val) + fd.r,
            iterations=int(sol.iterations),
            solve_time=float(sol.solve_time),
            native=native,
        )


class SCSAdapter(Adapter):
    name = "scs"
    package = "scs"
    default_tol = "1e-4"
    supports_soc = True
    supports_quadratic = True

    def _import(self) -> Any:
        import scs

        return scs

    def time_limit_settings(self, seconds: float) -> dict[str, Any]:
        return {"time_limit_secs": float(seconds)}

    def tolerance_settings(self, tol: float) -> dict[str, Any]:
        return {"eps_abs": tol, "eps_rel": tol}

    def _solve(self, inst: Instance, settings: dict[str, Any]) -> SolverResult:
        scs = self._import()
        fd = inst.float_data()
        rows = _cone_rows(inst)
        order = rows["zero"] + rows["nonneg"] + rows["soc"]  # SCS cone order
        perm = np.array(order, dtype=np.int64)
        A = sp.csc_matrix(fd.A[perm] if len(perm) else fd.A)
        data = {"P": upper_triangle(fd.P), "A": A, "b": fd.b[perm], "c": fd.q}
        cone = {"z": len(rows["zero"]), "l": len(rows["nonneg"]), "q": _soc_dims(inst)}
        solver = scs.SCS(data, cone, **{"verbose": False, **settings})
        sol = solver.solve()
        cones = [("zero", cone["z"]), ("nonneg", cone["l"])] + [("soc", d) for d in cone["q"]]
        native = NativeRun(
            form="conic",
            data={
                "P": data["P"],
                "P_triangle": "upper",
                "q": fd.q,
                "A": A,
                "b": data["b"],
                "cones": [c for c in cones if c[1]],
            },
            output={k: as_array(sol[k]) for k in ("x", "y", "s")},
            # SCS's Python API cannot read settings back: these are the C defaults
            # (glbopts.h) unless set explicitly.
            settings={"eps_abs": 1e-4, "eps_rel": 1e-4}
            | {k: settings[k] for k in ("eps_abs", "eps_rel") if k in settings},
            info={k: float(sol["info"][k]) for k in ("res_pri", "res_dual", "gap")},
        )
        info = sol["info"]
        raw = str(info["status"])
        status = {
            "solved": "optimal",
            "solved (inaccurate - reached max_iters)": "inaccurate",
            "solved (inaccurate - reached time_limit_secs)": "inaccurate",
            "infeasible": "primal_infeasible",
            "infeasible (inaccurate - reached max_iters)": "primal_infeasible",
            "unbounded": "dual_infeasible",
            "unbounded (inaccurate - reached max_iters)": "dual_infeasible",
        }.get(raw, "error")
        if status == "error" and raw.startswith("solved"):
            status = "inaccurate"
        y = as_array(sol["y"])
        z = None
        if y is not None:
            z = np.empty_like(y)
            z[perm] = y
        return self._result(
            status,
            raw,
            settings,
            x=as_array(sol["x"]),
            z=z,
            objective=float(info["pobj"]) + fd.r,
            iterations=int(info["iter"]),
            solve_time=float(info["solve_time"]) / 1000.0,
            native=native,
        )


class OSQPAdapter(Adapter):
    """OSQP on ``l <= A x <= u``: the boxed form (rows, then bounded columns) for MPS/QPS
    input, otherwise the conic rows (zero rows as equalities, nonneg rows as ``<= b``)."""

    name = "osqp"
    package = "osqp"
    default_tol = "1e-3"
    supports_soc = False
    supports_quadratic = True

    def _import(self) -> Any:
        import osqp

        return osqp

    def time_limit_settings(self, seconds: float) -> dict[str, Any]:
        return {"time_limit": float(seconds)}

    def tolerance_settings(self, tol: float) -> dict[str, Any]:
        return {"eps_abs": tol, "eps_rel": tol}

    def _solve(self, inst: Instance, settings: dict[str, Any]) -> SolverResult:
        osqp = self._import()
        fd = inst.float_data()
        boxed_cols: np.ndarray | None = None
        if inst.boxed is not None:
            bx = inst.boxed
            rl, ru, lb, ub = bx.float_bounds()
            boxed_cols = np.flatnonzero(np.isfinite(lb) | np.isfinite(ub))
            A_rows = bx.A.to_scipy()
            eye = sp.csr_array(
                (np.ones(len(boxed_cols)), (np.arange(len(boxed_cols)), boxed_cols)),
                shape=(len(boxed_cols), bx.n),
            )
            A = sp.csc_matrix(sp.vstack([A_rows, eye]))
            low = np.concatenate([rl, lb[boxed_cols]])
            up = np.concatenate([ru, ub[boxed_cols]])
            P, q = upper_triangle(bx.P.to_scipy()), fd.q
        else:
            rows = _cone_rows(inst)
            A = sp.csc_matrix(fd.A)
            low = fd.b.copy()
            low[rows["nonneg"]] = -np.inf
            up = fd.b.copy()
            P, q = upper_triangle(fd.P), fd.q
        solver = osqp.OSQP()
        solver.setup(P=_csc32(P), q=q, A=_csc32(A), l=low, u=up, **{"verbose": False, **settings})
        res = solver.solve(raise_error=False)
        st = solver._solver.get_settings()
        native = NativeRun(
            form="box",
            data={
                "P": P,
                "P_triangle": "upper",
                "q": q,
                "A": A,
                # OSQP clips the bounds to +-OSQP_INFTY on setup
                "l": np.maximum(low, -float(solver.constant("OSQP_INFTY"))),
                "u": np.minimum(up, float(solver.constant("OSQP_INFTY"))),
                "infinity": float(solver.constant("OSQP_INFTY")),
            },
            output={"x": as_array(res.x), "y": as_array(res.y)},
            settings={
                k: getattr(st, k)
                for k in (
                    "eps_abs",
                    "eps_rel",
                    "scaling",
                    "scaled_termination",
                    "check_dualgap",
                    "polishing",
                )
            },
            info={k: float(getattr(res.info, k)) for k in ("prim_res", "dual_res", "duality_gap")},
        )
        raw = str(res.info.status)
        status = {
            "solved": "optimal",
            "solved inaccurate": "inaccurate",
            "primal infeasible": "primal_infeasible",
            "primal infeasible inaccurate": "primal_infeasible",
            "dual infeasible": "dual_infeasible",
            "dual infeasible inaccurate": "dual_infeasible",
            "maximum iterations reached": "max_iter",
            "run time limit reached": "time_limit",
            "problem non convex": "error",
        }.get(raw, "error")
        x = as_array(res.x)
        y = as_array(res.y)
        z = None
        if y is not None and np.isfinite(y).all():
            if boxed_cols is not None and inst.boxed is not None and inst.origins is not None:
                m = inst.boxed.m
                y_col = np.zeros(inst.boxed.n)
                y_col[boxed_cols] = y[m:]
                z = boxed_duals_to_conic(inst.origins, y[:m], y_col)
            else:
                z = y
        obj = res.info.obj_val
        return self._result(
            status,
            raw,
            settings,
            x=x,
            z=z,
            objective=None if obj is None or not np.isfinite(obj) else float(obj) + fd.r,
            iterations=int(res.info.iter),
            solve_time=float(res.info.solve_time),
            native=native,
        )


class ECOSAdapter(Adapter):
    name = "ecos"
    package = "ecos"
    default_tol = "1e-8"
    supports_soc = True
    supports_quadratic = False

    def _import(self) -> Any:
        import ecos

        return ecos

    def tolerance_settings(self, tol: float) -> dict[str, Any]:
        return {"feastol": tol, "abstol": tol, "reltol": tol}

    def _solve(self, inst: Instance, settings: dict[str, Any]) -> SolverResult:
        ecos = self._import()
        fd = inst.float_data()
        rows = _cone_rows(inst)
        g_rows = np.array(rows["nonneg"] + rows["soc"], dtype=np.int64)
        e_rows = np.array(rows["zero"], dtype=np.int64)
        n = inst.problem.n
        G = sp.csc_matrix(fd.A[g_rows]) if len(g_rows) else sp.csc_matrix((0, n))
        h = fd.b[g_rows]
        dims = {"l": len(rows["nonneg"]), "q": _soc_dims(inst)}
        kw: dict[str, Any] = {"verbose": False, **settings}
        if len(e_rows):
            res = ecos.solve(fd.q, G, h, dims, sp.csc_matrix(fd.A[e_rows]), fd.b[e_rows], **kw)
        else:
            res = ecos.solve(fd.q, G, h, dims, **kw)
        info = res["info"]
        flag = int(info["exitFlag"])
        status = {
            0: "optimal",
            10: "inaccurate",
            1: "primal_infeasible",
            11: "primal_infeasible",
            2: "dual_infeasible",
            12: "dual_infeasible",
            -1: "max_iter",
            -2: "numerical_error",
            -3: "numerical_error",
            -7: "error",
        }.get(flag, "error")
        z = np.zeros(inst.problem.m)
        if len(g_rows):
            z[g_rows] = np.asarray(res["z"], dtype=np.float64)
        if len(e_rows):
            z[e_rows] = np.asarray(res["y"], dtype=np.float64)
        return self._result(
            status,
            f"{flag}: {info.get('infostring', '')}",
            settings,
            x=as_array(res["x"]),
            z=z,
            objective=float(info["pcost"]) + fd.r,
            iterations=int(info["iter"]),
            solve_time=float(info["timing"]["runtime"]),
        )


class HiGHSAdapter(Adapter):
    """HiGHS on the boxed form for MPS/QPS input, otherwise on the conic rows."""

    name = "highs"
    package = "highspy"
    default_tol = "1e-7"
    supports_soc = False
    supports_quadratic = True

    def _import(self) -> Any:
        import highspy

        return highspy

    def time_limit_settings(self, seconds: float) -> dict[str, Any]:
        return {"time_limit": float(seconds)}

    def tolerance_settings(self, tol: float) -> dict[str, Any]:
        # kkt_tolerance covers every feasibility and optimality measure once it differs
        # from its default; the IPM's gap tolerance has its own (tighter) default.
        return {
            "primal_feasibility_tolerance": tol,
            "dual_feasibility_tolerance": tol,
            "kkt_tolerance": tol,
            "ipm_optimality_tolerance": tol,
        }

    def _solve(self, inst: Instance, settings: dict[str, Any]) -> SolverResult:
        highspy = self._import()
        inf = highspy.kHighsInf
        fd = inst.float_data()
        if inst.boxed is not None:
            bx = inst.boxed
            rl, ru, lb, ub = bx.float_bounds()
            A = sp.csc_matrix(bx.A.to_scipy())
            P = bx.P.to_scipy()
            cost = fd.q
        else:
            rows = _cone_rows(inst)
            A = sp.csc_matrix(fd.A)
            ru = fd.b.copy()
            rl = fd.b.copy()
            rl[rows["nonneg"]] = -np.inf
            n = inst.problem.n
            lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
            P = fd.P
            cost = fd.q
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        for k, v in settings.items():
            st = h.setOptionValue(k, v)
            if st != highspy.HighsStatus.kOk:
                raise ValueError(f"HiGHS rejected option {k}={v!r}")
        lp = highspy.HighsLp()
        lp.num_col_ = A.shape[1]
        lp.num_row_ = A.shape[0]
        lp.col_cost_ = cost
        lp.col_lower_ = np.where(np.isfinite(lb), lb, -inf)
        lp.col_upper_ = np.where(np.isfinite(ub), ub, inf)
        lp.row_lower_ = np.where(np.isfinite(rl), rl, -inf)
        lp.row_upper_ = np.where(np.isfinite(ru), ru, inf)
        lp.offset_ = fd.r
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        h.passModel(lp)
        if P.nnz:
            lower = sp.csc_matrix(sp.tril(P))  # HiGHS takes the lower triangle by column
            hess = highspy.HighsHessian()
            hess.dim_ = lp.num_col_
            hess.format_ = highspy.HessianFormat.kTriangular
            hess.start_ = lower.indptr
            hess.index_ = lower.indices
            hess.value_ = lower.data
            if h.passHessian(hess) != highspy.HighsStatus.kOk:
                raise ValueError("HiGHS rejected the Hessian")
        h.run()
        ms = h.getModelStatus()
        raw = h.modelStatusToString(ms)
        status = {
            "Optimal": "optimal",
            "Infeasible": "primal_infeasible",
            "Unbounded": "dual_infeasible",
            "Primal infeasible or unbounded": "primal_infeasible",
            "Time limit reached": "time_limit",
            "Iteration limit reached": "max_iter",
        }.get(raw, "error")
        sol = h.getSolution()
        info = h.getInfo()
        native = NativeRun(
            form="highs",
            data={
                "P": sp.csc_matrix(sp.tril(P)) if P.nnz else sp.csc_matrix((lp.num_col_,) * 2),
                "P_triangle": "lower",
                "q": cost,
                "A": A,
                "col_lower": np.asarray(lp.col_lower_, dtype=np.float64),
                "col_upper": np.asarray(lp.col_upper_, dtype=np.float64),
                "row_lower": np.asarray(lp.row_lower_, dtype=np.float64),
                "row_upper": np.asarray(lp.row_upper_, dtype=np.float64),
            },
            output={
                k: np.asarray(getattr(sol, k), dtype=np.float64)
                for k in ("col_value", "row_value", "col_dual", "row_dual")
            }
            if sol.value_valid and sol.dual_valid
            else {},
            settings={
                k: h.getOptionValue(k)[1]
                for k in (
                    "primal_feasibility_tolerance",
                    "dual_feasibility_tolerance",
                    "kkt_tolerance",
                    "solver",
                )
            },
            info={
                "max_primal_infeasibility": float(info.max_primal_infeasibility),
                "max_dual_infeasibility": float(info.max_dual_infeasibility),
                "num_primal_infeasibilities": int(info.num_primal_infeasibilities),
                "num_dual_infeasibilities": int(info.num_dual_infeasibilities),
            },
        )
        x = as_array(list(sol.col_value)) if sol.value_valid else None
        z = None
        if sol.dual_valid:
            # HiGHS: P x + c - A'y - d = 0; the boxed convention negates both multipliers.
            y_row = -np.asarray(sol.row_dual, dtype=np.float64)
            y_col = -np.asarray(sol.col_dual, dtype=np.float64)
            if inst.boxed is not None and inst.origins is not None:
                z = boxed_duals_to_conic(inst.origins, y_row, y_col)
            else:
                z = y_row
        iters = sum(
            max(int(getattr(info, k, 0)), 0)
            for k in ("simplex_iteration_count", "ipm_iteration_count", "qp_iteration_count")
        )
        return self._result(
            status,
            raw,
            settings,
            x=x,
            z=z,
            objective=float(info.objective_function_value) if x is not None else None,
            iterations=iters,
            solve_time=float(h.getRunTime()),
            native=native,
        )


ADAPTERS: dict[str, Adapter] = {
    a.name: a
    for a in (ClarabelAdapter(), SCSAdapter(), OSQPAdapter(), ECOSAdapter(), HiGHSAdapter())
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown solver {name!r}; choose from {sorted(ADAPTERS)}") from None


__all__ = [
    "ADAPTERS",
    "ClarabelAdapter",
    "ECOSAdapter",
    "HiGHSAdapter",
    "OSQPAdapter",
    "SCSAdapter",
    "get_adapter",
]
