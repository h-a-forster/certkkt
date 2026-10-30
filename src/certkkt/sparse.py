"""Exact sparse linear algebra over the rationals.

:class:`QMatrix` is a CSR matrix whose entries are exact rationals (:data:`certkkt.rational.Q`).
:func:`solve_any` finds one solution of a (possibly rectangular or singular) consistent
system by sparse Gaussian elimination with a Markowitz-style pivot rule, and
:func:`psd_check` decides positive semidefiniteness exactly by symmetric elimination.

Exact elimination has no stability concerns, but fill-in and coefficient growth can make it
expensive, so both routines take an operation budget and raise :class:`BudgetExceeded`
instead of running unboundedly.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from certkkt.rational import ZERO, Q, to_q

#: Default number of scalar update operations allowed in one elimination.
DEFAULT_BUDGET = 20_000_000


class BudgetExceeded(RuntimeError):
    """An exact computation needed more operations or time than its budget allowed."""


def check_deadline(deadline: float | None) -> None:
    """Raise BudgetExceeded once ``time.perf_counter()`` passes ``deadline``."""
    if deadline is not None and time.perf_counter() > deadline:
        raise BudgetExceeded("time budget exhausted")


class QMatrix:
    """Immutable CSR matrix of exact rationals. Explicit zeros are never stored."""

    __slots__ = ("shape", "indptr", "indices", "data")

    def __init__(
        self,
        shape: tuple[int, int],
        indptr: Sequence[int],
        indices: Sequence[int],
        data: Sequence[Any],
    ) -> None:
        m, n = shape
        if m < 0 or n < 0:
            raise ValueError(f"invalid shape {shape}")
        if len(indptr) != m + 1 or indptr[0] != 0 or indptr[-1] != len(indices):
            raise ValueError("inconsistent CSR index pointer")
        if len(indices) != len(data):
            raise ValueError("indices and data differ in length")
        for k in range(m):
            if indptr[k] > indptr[k + 1]:
                raise ValueError("CSR index pointer is not monotone")
        if indices and (min(indices) < 0 or max(indices) >= n):
            raise ValueError("column index out of range")
        self.shape = (int(m), int(n))
        self.indptr = list(indptr)
        self.indices = list(indices)
        self.data = list(data)

    # -- construction -------------------------------------------------------------------
    @classmethod
    def from_triplets(
        cls,
        shape: tuple[int, int],
        rows: Iterable[int],
        cols: Iterable[int],
        vals: Iterable[object],
    ) -> QMatrix:
        """Build from (row, col, value) triplets; duplicates are summed, zeros dropped."""
        m, n = shape
        acc: list[dict[int, Any]] = [{} for _ in range(m)]
        for i, j, v in zip(rows, cols, vals, strict=True):
            i, j = int(i), int(j)
            if not (0 <= i < m and 0 <= j < n):
                raise ValueError(f"entry ({i}, {j}) outside shape {shape}")
            q = to_q(v)
            row = acc[i]
            row[j] = row.get(j, ZERO) + q
        return cls.from_rows(shape, acc)

    @classmethod
    def from_rows(cls, shape: tuple[int, int], rows: Sequence[Mapping[int, Any]]) -> QMatrix:
        """Build from one ``{col: value}`` mapping per row (values already rational)."""
        if len(rows) != shape[0]:
            raise ValueError("row count does not match shape")
        indptr = [0]
        indices: list[int] = []
        data: list[Any] = []
        for row in rows:
            for j in sorted(row):
                v = row[j]
                if v:
                    indices.append(j)
                    data.append(v)
            indptr.append(len(indices))
        return cls(shape, indptr, indices, data)

    @classmethod
    def from_scipy(cls, mat: sp.spmatrix | sp.sparray | np.ndarray) -> QMatrix:
        """Exact copy of a float matrix (each float converts to the rational it denotes)."""
        csr = sp.csr_array(mat)
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        data = [to_q(v) for v in csr.data.tolist()]
        return cls(csr.shape, csr.indptr.tolist(), csr.indices.tolist(), data)

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> QMatrix:
        return cls(shape, [0] * (shape[0] + 1), [], [])

    @classmethod
    def identity(cls, n: int, scale: object = 1) -> QMatrix:
        q = to_q(scale)
        if not q:
            return cls.zeros((n, n))
        return cls((n, n), list(range(n + 1)), list(range(n)), [q] * n)

    # -- queries ------------------------------------------------------------------------
    @property
    def nnz(self) -> int:
        return len(self.data)

    def row(self, i: int) -> tuple[list[int], list[Any]]:
        a, b = self.indptr[i], self.indptr[i + 1]
        return self.indices[a:b], self.data[a:b]

    def row_dicts(self) -> list[dict[int, Any]]:
        out = []
        for i in range(self.shape[0]):
            cols, vals = self.row(i)
            out.append(dict(zip(cols, vals, strict=True)))
        return out

    def matvec(self, x: Sequence[Any]) -> list[Any]:
        """Exact ``A @ x``."""
        if len(x) != self.shape[1]:
            raise ValueError(f"x has length {len(x)}, expected {self.shape[1]}")
        out = []
        ind, dat, ptr = self.indices, self.data, self.indptr
        for i in range(self.shape[0]):
            total = ZERO
            for k in range(ptr[i], ptr[i + 1]):
                xv = x[ind[k]]
                if xv:
                    total += dat[k] * xv
            out.append(total)
        return out

    def rmatvec(self, y: Sequence[Any]) -> list[Any]:
        """Exact ``A.T @ y``."""
        if len(y) != self.shape[0]:
            raise ValueError(f"y has length {len(y)}, expected {self.shape[0]}")
        out = [ZERO] * self.shape[1]
        ind, dat, ptr = self.indices, self.data, self.indptr
        for i in range(self.shape[0]):
            yv = y[i]
            if not yv:
                continue
            for k in range(ptr[i], ptr[i + 1]):
                out[ind[k]] += dat[k] * yv
        return out

    def transpose(self) -> QMatrix:
        m, n = self.shape
        rows: list[dict[int, Any]] = [{} for _ in range(n)]
        for i in range(m):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                rows[self.indices[k]][i] = self.data[k]
        return QMatrix.from_rows((n, m), rows)

    def select_rows(self, idx: Sequence[int]) -> QMatrix:
        indptr = [0]
        indices: list[int] = []
        data: list[Any] = []
        for i in idx:
            a, b = self.indptr[i], self.indptr[i + 1]
            indices.extend(self.indices[a:b])
            data.extend(self.data[a:b])
            indptr.append(len(indices))
        return QMatrix((len(idx), self.shape[1]), indptr, indices, data)

    def to_scipy(self) -> sp.csr_array:
        """Float copy, each entry correctly rounded to the nearest double."""
        data = np.array(
            [int(v.numerator) / int(v.denominator) for v in self.data], dtype=np.float64
        )
        itype = np.int32 if self.nnz < 2**31 - 1 else np.int64  # solvers expect 32-bit
        return sp.csr_array(
            (data, np.array(self.indices, dtype=itype), np.array(self.indptr, dtype=itype)),
            shape=self.shape,
        )

    def is_symmetric(self) -> bool:
        if self.shape[0] != self.shape[1]:
            return False
        rows = self.row_dicts()
        for i, row in enumerate(rows):
            for j, v in row.items():
                if rows[j].get(i) != v:
                    return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QMatrix):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.indptr == other.indptr
            and self.indices == other.indices
            and self.data == other.data
        )

    def __repr__(self) -> str:
        return f"QMatrix(shape={self.shape}, nnz={self.nnz})"


def vstack(blocks: Sequence[QMatrix]) -> QMatrix:
    if not blocks:
        raise ValueError("nothing to stack")
    n = blocks[0].shape[1]
    indptr = [0]
    indices: list[int] = []
    data: list[Any] = []
    m = 0
    for blk in blocks:
        if blk.shape[1] != n:
            raise ValueError("column counts differ")
        base = len(indices)
        indices.extend(blk.indices)
        data.extend(blk.data)
        indptr.extend(base + p for p in blk.indptr[1:])
        m += blk.shape[0]
    return QMatrix((m, n), indptr, indices, data)


# -- exact elimination ----------------------------------------------------------------------


#: Prime used to find the rank profile of a system without coefficient growth.
MODULUS = 2**61 - 1


def _eliminate(
    work: list[dict[int, Any]],
    b: list[Any],
    penalty: Mapping[int, int],
    budget: int,
    modulus: int | None,
    deadline: float | None = None,
) -> list[tuple[int, int]] | None:
    """Sparse Gaussian elimination in place; returns the (row, col) pivots in order, or
    None if a zero row has a nonzero right-hand side. Over the rationals when ``modulus``
    is None, otherwise over the integers modulo that prime."""
    col_rows: dict[int, set[int]] = {}
    for i, row in enumerate(work):
        for j in row:
            col_rows.setdefault(j, set()).add(i)
    active = set(range(len(work)))
    heap = [(len(row), i) for i, row in enumerate(work)]
    heapq.heapify(heap)
    pivots: list[tuple[int, int]] = []
    ops = 0
    p = modulus
    while active:
        # Collect up to three current shortest active rows (lazy heap entries).
        cands: list[int] = []
        while heap and len(cands) < 3:
            length, i = heapq.heappop(heap)
            if i in active and len(work[i]) == length:
                cands.append(i)
        if not cands:
            break
        best: tuple[tuple[int, int], int, int] | None = None
        for i in cands:
            row = work[i]
            for j in row:
                cost = (penalty.get(j, 0), (len(row) - 1) * (len(col_rows[j]) - 1))
                if best is None or cost < best[0]:
                    best = (cost, i, j)
        for i in cands:
            if not work[i]:  # empty row: a consistency condition
                active.discard(i)
                if b[i]:
                    return None
        if best is None:
            continue
        _, r, c = best
        for i in cands:
            if i != r and i in active:
                heapq.heappush(heap, (len(work[i]), i))
        prow = work[r]
        pval = prow[c]
        inv = pow(pval, -1, p) if p else None
        active.discard(r)
        for j in prow:
            col_rows[j].discard(r)
        for i in list(col_rows[c]):
            row = work[i]
            f = row[c] * inv % p if p else row[c] / pval
            for j, v in prow.items():
                new = (row.get(j, 0) - f * v) % p if p else row.get(j, ZERO) - f * v
                if new:
                    if j not in row:
                        col_rows[j].add(i)
                    row[j] = new
                elif j in row:
                    del row[j]
                    col_rows[j].discard(i)
            if b[r]:
                b[i] = (b[i] - f * b[r]) % p if p else b[i] - f * b[r]
            ops += len(prow)
            heapq.heappush(heap, (len(row), i))
        if ops > budget:
            raise BudgetExceeded(f"exact elimination exceeded {budget} operations")
        if deadline is not None and len(pivots) % 64 == 0:
            check_deadline(deadline)
        pivots.append((r, c))
    for i in active:
        if not work[i] and b[i]:
            return None
    return pivots


def _copy_rows(rows: Sequence[Mapping[int, Any]], ncols: int) -> list[dict[int, Any]]:
    work = []
    for row in rows:
        d = {}
        for j, v in row.items():
            if not 0 <= j < ncols:
                raise ValueError(f"column {j} out of range")
            if v:
                d[j] = v
        work.append(d)
    return work


def solve_any(
    rows: Sequence[Mapping[int, Any]],
    rhs: Sequence[Any],
    ncols: int,
    *,
    col_penalty: Mapping[int, int] | None = None,
    budget: int = DEFAULT_BUDGET,
    deadline: float | None = None,
) -> list[Any] | None:
    """One exact solution of ``M x = rhs``, or None if the system is inconsistent.

    ``rows[i]`` maps column index to the (nonzero, rational) entry ``M[i, col]``. The
    system may be rectangular or rank deficient; variables that never become pivots are
    set to zero, so the solution is supported on the pivot columns. ``col_penalty`` (lower
    is preferred) steers which columns become pivots, ahead of the Markowitz fill
    estimate. Raises BudgetExceeded after ``budget`` scalar updates or past ``deadline``
    (a ``time.perf_counter()`` value).

    Exact elimination over the rationals; :func:`solve_sparse` is usually much faster.
    """
    if len(rows) != len(rhs):
        raise ValueError("rows and rhs differ in length")
    work = _copy_rows(rows, ncols)
    b = list(rhs)
    pivots = _eliminate(work, b, col_penalty or {}, budget, None, deadline)
    if pivots is None:
        return None
    x = [ZERO] * ncols
    for r, c in reversed(pivots):
        row = work[r]
        total = b[r]
        for j, v in row.items():
            if j != c:
                xv = x[j]
                if xv:
                    total -= v * xv
        x[c] = total / row[c]
    return x


def _mod(v: Any, p: int) -> int:
    den = int(v.denominator) % p
    if den == 0:
        raise ZeroDivisionError("denominator divisible by the modulus")
    return int(v.numerator) * pow(den, -1, p) % p


def solve_sparse(
    rows: Sequence[Mapping[int, Any]],
    rhs: Sequence[Any],
    ncols: int,
    *,
    col_penalty: Mapping[int, int] | None = None,
    budget: int = DEFAULT_BUDGET,
    deadline: float | None = None,
) -> list[Any] | None:
    """Same contract as :func:`solve_any`, computed without coefficient growth.

    1. Elimination modulo a large prime finds a rank profile: pivot rows R and columns C
       such that ``M[R, C]`` is (with overwhelming probability) nonsingular over Q.
    2. ``M[R, C] y = rhs[R]`` is solved exactly by :func:`solve_refined`.
    3. ``x[C] = y``, zero elsewhere, is verified against every row in exact arithmetic.

    Any failure (unlucky prime, float factorisation failure, stalled refinement) falls
    back to exact elimination. A None (inconsistent) answer from step 1 is trusted: it
    only ever leads to a weaker certificate, never to a wrong one.
    """
    if len(rows) != len(rhs):
        raise ValueError("rows and rhs differ in length")
    work = _copy_rows(rows, ncols)
    try:
        wmod = [{j: _mod(v, MODULUS) for j, v in row.items()} for row in work]
        wmod = [{j: v for j, v in row.items() if v} for row in wmod]
        bmod = [_mod(v, MODULUS) for v in rhs]
    except ZeroDivisionError:
        return solve_any(
            work, rhs, ncols, col_penalty=col_penalty, budget=budget, deadline=deadline
        )
    pivots = _eliminate(wmod, bmod, col_penalty or {}, budget, MODULUS, deadline)
    if pivots is None:
        return None
    if not pivots:
        return [ZERO] * ncols if not any(rhs) else None
    prow = [r for r, _ in pivots]
    pcol = [c for _, c in pivots]
    cpos = {c: k for k, c in enumerate(pcol)}
    sub_rows = []
    for r in prow:
        sub_rows.append({cpos[j]: v for j, v in work[r].items() if j in cpos})
    k = len(pivots)
    y = solve_refined(
        QMatrix.from_rows((k, k), sub_rows), [rhs[r] for r in prow], deadline=deadline
    )
    if y is not None:
        x = [ZERO] * ncols
        for c, v in zip(pcol, y, strict=True):
            x[c] = v
        ok = True
        for row, bi in zip(work, rhs, strict=True):
            total = ZERO
            for j, v in row.items():
                if x[j]:
                    total += v * x[j]
            if total != bi:
                ok = False
                break
        if ok:
            return x
    return solve_any(work, rhs, ncols, col_penalty=col_penalty, budget=budget, deadline=deadline)


def solve_refined(
    mat: QMatrix,
    rhs: Sequence[Any],
    *,
    max_iter: int = 60,
    min_gain_bits: int = 8,
    deadline: float | None = None,
) -> list[Any] | None:
    """Exact solution of a square nonsingular system by iterative refinement and rational
    reconstruction, or None if the float factorisation fails or refinement stalls.

    Each step solves for the correction with a float sparse LU factorisation of ``mat``,
    adds it exactly, and recomputes the residual exactly. After every step each
    coordinate is replaced by the simplest rational within a small multiple of the last
    correction (at steps 1, 2, 4, 8, ... and the last); if that candidate satisfies the
    system exactly it is returned. This finds
    solutions with modest denominators far faster than exact elimination, whose
    intermediate coefficients grow (the approach of Wan, and of Gleixner, Steffy and
    Wolter for exact LP). Nothing returned is unverified.
    """
    import scipy.sparse.linalg as spla

    from certkkt.rational import max_abs, simplest_between

    n = mat.shape[0]
    if mat.shape[1] != n or len(rhs) != n:
        raise ValueError("solve_refined needs a square system")
    if n == 0:
        return []
    try:
        lu = spla.splu(sp.csc_matrix(mat.to_scipy()))
    except RuntimeError:  # exactly singular in floating point
        return None
    x = [ZERO] * n
    r = list(rhs)
    prev = max_abs(r)
    if prev == 0:
        return x
    for it in range(1, max_iter + 1):
        check_deadline(deadline)
        exp = _floor_log2(prev)
        scale = Q(2) ** exp
        rf = np.array([_nearest_float(v / scale) for v in r])
        d = lu.solve(rf)
        if not np.isfinite(d).all():
            return None
        dq = [to_q(v) * scale if v else ZERO for v in d.tolist()]
        x = [a + b if b else a for a, b in zip(x, dq, strict=True)]
        ax = mat.matvec(x)
        r = [bi - ai for bi, ai in zip(rhs, ax, strict=True)]
        cur = max_abs(r)
        if cur == 0:
            return x
        # Rational reconstruction with a tolerance well below the last correction.
        tol = max_abs(dq) / Q(2) ** 20
        if tol and (it & (it - 1) == 0 or it == max_iter):
            cand = [simplest_between(v - tol, v + tol) for v in x]
            ac = mat.matvec(cand)
            if all(bi == ai for bi, ai in zip(rhs, ac, strict=True)):
                return cand
        if cur * Q(2) ** min_gain_bits > prev:  # stalled: ill-conditioned or singular
            return None
        prev = cur
    return None


def _floor_log2(v: Any) -> int:
    """floor(log2(v)) for a positive rational."""
    num, den = int(v.numerator), int(v.denominator)
    e = num.bit_length() - den.bit_length()
    if (num << max(-e, 0)) < (den << max(e, 0)):
        e -= 1
    return e


def _nearest_float(v: Any) -> float:
    return int(v.numerator) / int(v.denominator)


@dataclass(frozen=True)
class PSDResult:
    """Outcome of :func:`psd_check`: ``status`` is "psd", "not_psd" or "unknown"."""

    status: str
    reason: str = ""

    @property
    def is_psd(self) -> bool:
        return self.status == "psd"


def psd_check(mat: QMatrix, *, budget: int = DEFAULT_BUDGET) -> PSDResult:
    """Decide exactly whether a symmetric rational matrix is positive semidefinite.

    Symmetric Gaussian elimination: a matrix is PSD iff, after eliminating a positive
    diagonal pivot, its Schur complement is PSD; a negative diagonal, or a zero diagonal
    with a nonzero off-diagonal in its row, refutes it. Pivots are chosen by smallest
    degree. Returns status "unknown" if the budget runs out.
    """
    n = mat.shape[0]
    if mat.shape[1] != n:
        raise ValueError("matrix is not square")
    if not mat.is_symmetric():
        raise ValueError("matrix is not symmetric")
    rows = mat.row_dicts()
    # Quick exact sufficient test: nonnegative diagonal and weak diagonal dominance.
    dominant = True
    for i, row in enumerate(rows):
        d = row.get(i, ZERO)
        if d < 0:
            return PSDResult("not_psd", f"negative diagonal entry at {i}")
        off = sum((abs(v) for j, v in row.items() if j != i), ZERO)
        if off > d:
            dominant = False
            if d == 0:
                return PSDResult("not_psd", f"zero diagonal with nonzero off-diagonal at {i}")
    if dominant:
        return PSDResult("psd", "diagonally dominant with nonnegative diagonal")

    remaining = {i for i in range(n) if rows[i]}
    heap = [(len(rows[i]), i) for i in remaining]
    heapq.heapify(heap)
    ops = 0
    while remaining:
        pivot = None
        while heap:
            deg, i = heapq.heappop(heap)
            if i in remaining and len(rows[i]) == deg:
                pivot = i
                break
        if pivot is None:
            break
        row = rows[pivot]
        d = row.get(pivot, ZERO)
        if d < 0:
            return PSDResult("not_psd", f"negative pivot at {pivot}")
        if d == 0:
            if any(j != pivot for j in row):
                return PSDResult("not_psd", f"zero pivot with nonzero off-diagonal at {pivot}")
            remaining.discard(pivot)
            continue
        remaining.discard(pivot)
        nbrs = [(j, v) for j, v in row.items() if j != pivot]
        for j, _ in nbrs:
            del rows[j][pivot]
        for a, (j, vj) in enumerate(nbrs):
            fj = vj / d
            rj = rows[j]
            for k, vk in nbrs[a:]:
                new = rj.get(k, ZERO) - fj * vk
                if new:
                    rj[k] = new
                    if k != j:
                        rows[k][j] = new
                else:
                    rj.pop(k, None)
                    if k != j:
                        rows[k].pop(j, None)
            ops += len(nbrs) - a
            heapq.heappush(heap, (len(rj), j))
        rows[pivot] = {}
        if ops > budget:
            return PSDResult("unknown", f"elimination exceeded {budget} operations")
    return PSDResult("psd", "exact symmetric elimination")


def q_list(values: Iterable[object]) -> list[Any]:
    """Exact rationals from floats, ints or decimal strings."""
    return [to_q(v) for v in values]


__all__ = [
    "DEFAULT_BUDGET",
    "BudgetExceeded",
    "check_deadline",
    "PSDResult",
    "Q",
    "QMatrix",
    "psd_check",
    "q_list",
    "solve_any",
    "solve_refined",
    "solve_sparse",
    "vstack",
]
