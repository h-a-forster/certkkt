"""Reader for the Conic Benchmark Format (CBF, versions 1 to 3) used by CBLIB.

Supported: linear and quadratic cones (``F``, ``L+``, ``L-``, ``L=``, ``Q``, ``QR``) on
variables and constraints, scalar objective and constraint coefficients. PSD variables or
constraints, exponential and power cones, and integer variables are rejected loudly.

CBF states ``A x + b in K`` for constraints and ``x in K`` for variables. Both become
canonical rows ``s = T (A x + b)`` with ``s`` in a zero, nonneg or SOC cone, where ``T`` is
the identity, a sign flip (``L-``) or the rational map that turns a rotated cone
``2 x0 x1 >= ||w||^2, x0, x1 >= 0`` into the second-order cone
``(x0 + x1/2, x0 - x1/2, w)`` (exact: no square roots). Numbers are exact decimals.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from certkkt.problem import Cone, ConicProblem, merge_cones
from certkkt.qps import FormatError
from certkkt.rational import ONE, ZERO, Q, as_double, to_q
from certkkt.sparse import QMatrix

_CONE_MAP = {"F": "free", "L+": "nonneg", "L-": "nonpos", "L=": "zero", "Q": "soc", "QR": "rsoc"}
_HALF = Q(1, 2)


def read_cbf(path: str | Path, *, exact_decimals: bool = True) -> ConicProblem:
    """Read a CBF file (optionally gzip-compressed) into an exact ConicProblem.

    ``exact_decimals=False`` rounds every number to the nearest double first.
    """
    p = Path(path)
    raw = p.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return parse_cbf(
        raw.decode("latin-1"), name=p.name.split(".")[0], exact_decimals=exact_decimals
    )


def _lines(text: str) -> Iterator[tuple[int, list[str]]]:
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if s and not s.startswith("#"):
            yield lineno, s.split()


def parse_cbf(text: str, name: str = "problem", *, exact_decimals: bool = True) -> ConicProblem:
    it = _lines(text)
    sense = 1
    var_cones: list[tuple[str, int]] | None = None
    con_cones: list[tuple[str, int]] = []
    n = 0
    m = 0
    obj: dict[int, Any] = {}
    obj_b = ZERO
    a_entries: dict[tuple[int, int], Any] = {}
    b_entries: dict[int, Any] = {}

    def next_tokens(what: str) -> tuple[int, list[str]]:
        try:
            return next(it)
        except StopIteration:
            raise FormatError(f"unexpected end of file in {what}") from None

    def count(what: str) -> int:
        ln, t = next_tokens(what)
        try:
            return int(t[0])
        except ValueError:
            raise FormatError(f"line {ln}: expected a count in {what}") from None

    def cone_list(what: str) -> tuple[int, list[tuple[str, int]]]:
        ln, t = next_tokens(what)
        if len(t) != 2:
            raise FormatError(f"line {ln}: expected 'size ncones' in {what}")
        total, k = int(t[0]), int(t[1])
        cones = []
        for _ in range(k):
            ln2, ct = next_tokens(what)
            if len(ct) != 2:
                raise FormatError(f"line {ln2}: expected 'cone dim' in {what}")
            kind = _CONE_MAP.get(ct[0])
            if kind is None:
                raise FormatError(f"line {ln2}: unsupported cone {ct[0]!r}")
            dim = int(ct[1])
            if dim < 1:
                raise FormatError(f"line {ln2}: cone dimension must be positive")
            if kind == "rsoc" and dim < 2:
                raise FormatError(f"line {ln2}: rotated cone of dimension {dim} < 2")
            cones.append((kind, dim))
        if sum(d for _, d in cones) != total:
            raise FormatError(f"{what}: cone dimensions do not add up to {total}")
        return total, cones

    def num(tok: str, ln: int) -> Any:
        try:
            return to_q(tok) if exact_decimals else as_double(to_q(tok))
        except ValueError:
            raise FormatError(f"line {ln}: bad number {tok!r}") from None

    for ln, t in it:
        key = t[0].upper()
        if key == "VER":
            _, v = next_tokens("VER")
            if int(v[0]) not in (1, 2, 3):
                raise FormatError(f"unsupported CBF version {v[0]}")
        elif key == "OBJSENSE":
            _, v = next_tokens("OBJSENSE")
            if v[0].upper() not in ("MIN", "MAX"):
                raise FormatError(f"line {ln}: bad objective sense")
            sense = 1 if v[0].upper() == "MIN" else -1
        elif key == "VAR":
            n, var_cones = cone_list("VAR")
        elif key == "CON":
            m, con_cones = cone_list("CON")
        elif key == "INT":
            if count("INT") > 0:
                raise FormatError("integer variables are not supported (continuous only)")
        elif key == "OBJACOORD":
            for _ in range(count("OBJACOORD")):
                ln2, e = next_tokens("OBJACOORD")
                j = int(e[0])
                if not 0 <= j < n:
                    raise FormatError(f"line {ln2}: variable index out of range")
                obj[j] = obj.get(j, ZERO) + num(e[1], ln2)
        elif key == "OBJBCOORD":
            ln2, e = next_tokens("OBJBCOORD")
            obj_b = num(e[0], ln2)
        elif key == "ACOORD":
            for _ in range(count("ACOORD")):
                ln2, e = next_tokens("ACOORD")
                i, j = int(e[0]), int(e[1])
                if not (0 <= i < m and 0 <= j < n):
                    raise FormatError(f"line {ln2}: ACOORD index out of range")
                a_entries[(i, j)] = a_entries.get((i, j), ZERO) + num(e[2], ln2)
        elif key == "BCOORD":
            for _ in range(count("BCOORD")):
                ln2, e = next_tokens("BCOORD")
                i = int(e[0])
                if not 0 <= i < m:
                    raise FormatError(f"line {ln2}: BCOORD index out of range")
                b_entries[i] = b_entries.get(i, ZERO) + num(e[1], ln2)
        elif key in ("PSDVAR", "PSDCON", "FCOORD", "OBJFCOORD", "HCOORD", "DCOORD"):
            if count(key) > 0:
                raise FormatError(f"{key}: semidefinite data is not supported")
        elif key in ("POWCONES", "POW*CONES", "CHANGE"):
            raise FormatError(f"{key} is not supported")
        else:
            raise FormatError(f"line {ln}: unknown keyword {t[0]!r}")

    if var_cones is None:
        raise FormatError("missing VAR section")

    con_rows: list[dict[int, Any]] = [{} for _ in range(m)]
    for (i, j), v in a_entries.items():
        if v:
            con_rows[i][j] = v
    rows: list[dict[int, Any]] = []
    rhs: list[Any] = []
    cones: list[Cone] = []

    def emit(kind: str, block: list[dict[int, Any]], bvals: list[Any]) -> None:
        # Block rows hold M (with s = T(M x + c)); canonical row is -T M, rhs T c.
        if kind == "free":
            return
        if kind == "rsoc":
            a0, a1, rest = block[0], block[1], block[2:]
            c0, c1, crest = bvals[0], bvals[1], bvals[2:]
            block = [_comb(a0, ONE, a1, _HALF), _comb(a0, ONE, a1, -_HALF), *rest]
            bvals = [c0 + c1 * _HALF, c0 - c1 * _HALF, *crest]
            kind = "soc"
        if kind == "soc" and len(block) == 1:  # a one-dimensional SOC is t >= 0
            kind = "nonneg"
        sign = -ONE if kind == "nonpos" else ONE
        ck = "nonneg" if kind == "nonpos" else kind
        for row, c in zip(block, bvals, strict=True):
            rows.append({j: -sign * v for j, v in row.items()})
            rhs.append(sign * c)
        cones.append(Cone(ck, len(block)))

    start = 0
    for kind, dim in con_cones:
        emit(
            kind,
            con_rows[start : start + dim],
            [b_entries.get(i, ZERO) for i in range(start, start + dim)],
        )
        start += dim
    start = 0
    for kind, dim in var_cones:
        emit(kind, [{j: ONE} for j in range(start, start + dim)], [ZERO] * dim)
        start += dim

    A = QMatrix.from_rows((len(rows), n), rows)
    q = [obj.get(j, ZERO) for j in range(n)]
    r = obj_b
    if sense == -1:
        q = [-v for v in q]
        r = -r
    return ConicProblem(
        P=QMatrix.zeros((n, n)),
        q=q,
        A=A,
        b=rhs,
        cones=merge_cones(cones),
        r=r,
        name=name,
        objective_sign=sense,
    )


def _comb(a: dict[int, Any], ca: Any, b: dict[int, Any], cb: Any) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for j, v in a.items():
        out[j] = ca * v
    for j, v in b.items():
        new = out.get(j, ZERO) + cb * v
        if new:
            out[j] = new
        else:
            out.pop(j, None)
    return out


__all__ = ["parse_cbf", "read_cbf"]
