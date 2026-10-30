"""Reader for MPS and QPS files (free format), keeping every number exact.

Numbers are parsed as exact decimals (``"0.1"`` is ``1/10``), so the certified problem is
the one written in the file, not its binary rounding. Supported sections: NAME, OBJSENSE,
ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUADOBJ (one triangle), QMATRIX (full), ENDATA.
Conventions follow the common MPS reading (as in HiGHS and CPLEX):

* the first N row is the objective; other N rows are dropped;
* an RHS entry on the objective row is the negated objective constant;
* RANGES on an E row widens it upward for R > 0 and downward for R < 0;
* default column bounds are [0, +inf); an UP bound with a negative value on a column whose
  lower bound is still the default 0 sets the lower bound to -inf (recorded as a warning);
* QUADOBJ / QMATRIX define ``1/2 x'Px``.

Integer markers and integer bound types are rejected: certkkt certifies continuous
problems only.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any

from certkkt.problem import BoxedQP
from certkkt.rational import ZERO, Q, as_double, to_q
from certkkt.sparse import QMatrix


class FormatError(ValueError):
    """The file is malformed or uses an unsupported feature."""


_SECTIONS = {
    "NAME",
    "OBJSENSE",
    "ROWS",
    "COLUMNS",
    "RHS",
    "RANGES",
    "BOUNDS",
    "QUADOBJ",
    "QSECTION",
    "QMATRIX",
    "ENDATA",
}


def read_qps(path: str | Path, *, exact_decimals: bool = True) -> BoxedQP:
    """Read an MPS/QPS file (optionally gzip-compressed) into an exact BoxedQP.

    ``exact_decimals=False`` rounds every number to the nearest double first.
    """
    p = Path(path)
    raw = p.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return parse_qps(
        raw.decode("latin-1"), default_name=p.stem.split(".")[0], exact_decimals=exact_decimals
    )


def parse_qps(text: str, default_name: str = "problem", *, exact_decimals: bool = True) -> BoxedQP:
    name = default_name
    sense = 1
    obj_row: str | None = None
    row_index: dict[str, int] = {}
    row_type: list[str] = []
    row_names: list[str] = []
    dropped_n: set[str] = set()
    col_index: dict[str, int] = {}
    col_names: list[str] = []
    entries: dict[tuple[int, int], Any] = {}
    cost: dict[int, Any] = {}
    rhs: dict[int, Any] = {}
    c0 = ZERO
    ranges: dict[int, Any] = {}
    lb: dict[int, Any | None] = {}
    ub: dict[int, Any | None] = {}
    neg_up_warn: list[str] = []
    quad: dict[tuple[int, int], Any] = {}
    quad_kind: str | None = None
    rhs_set: str | None = None
    range_set: str | None = None
    bound_set: str | None = None
    section: str | None = None
    seen_end = False

    def col(nm: str) -> int:
        j = col_index.get(nm)
        if j is None:
            raise FormatError(f"unknown column {nm!r}")
        return j

    def num(tok: str, lineno: int) -> Any:
        try:
            return to_q(tok) if exact_decimals else as_double(to_q(tok))
        except ValueError:
            raise FormatError(f"line {lineno}: bad number {tok!r}") from None

    for lineno, line in enumerate(io.StringIO(text), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        tokens = stripped.split()
        if not line[0].isspace():
            head = tokens[0].upper()
            if head not in _SECTIONS:
                raise FormatError(f"line {lineno}: unknown section {tokens[0]!r}")
            section = head
            if head == "NAME":
                name = tokens[1] if len(tokens) > 1 else default_name
            elif head == "OBJSENSE" and len(tokens) > 1:
                sense = _sense(tokens[1], lineno)
            elif head in ("QUADOBJ", "QSECTION", "QMATRIX"):
                kind = "full" if head == "QMATRIX" else "triangle"
                if quad_kind is not None and quad_kind != kind:
                    raise FormatError("both QUADOBJ and QMATRIX sections present")
                quad_kind = kind
                if head == "QSECTION" and len(tokens) > 1 and tokens[1] != obj_row:
                    raise FormatError("QSECTION for a constraint row (QCQP) is not supported")
            elif head == "ENDATA":
                seen_end = True
                break
            continue

        if section == "OBJSENSE":
            sense = _sense(tokens[0], lineno)
        elif section == "ROWS":
            if len(tokens) != 2:
                raise FormatError(f"line {lineno}: expected 'type name' in ROWS")
            t, nm = tokens[0].upper(), tokens[1]
            if t == "N":
                if obj_row is None:
                    obj_row = nm
                else:
                    dropped_n.add(nm)
                continue
            if t not in ("E", "L", "G"):
                raise FormatError(f"line {lineno}: unknown row type {t!r}")
            if nm in row_index or nm == obj_row:
                raise FormatError(f"line {lineno}: duplicate row {nm!r}")
            row_index[nm] = len(row_type)
            row_type.append(t)
            row_names.append(nm)
        elif section == "COLUMNS":
            if len(tokens) >= 3 and tokens[1].strip("'").upper() == "MARKER":
                raise FormatError("integer markers are not supported (continuous problems only)")
            if len(tokens) not in (3, 5):
                raise FormatError(f"line {lineno}: expected 'col row value [row value]'")
            cn = tokens[0]
            j = col_index.get(cn)
            if j is None:
                j = len(col_names)
                col_index[cn] = j
                col_names.append(cn)
            for k in range(1, len(tokens), 2):
                rn, v = tokens[k], num(tokens[k + 1], lineno)
                if rn == obj_row:
                    if j in cost:
                        raise FormatError(f"line {lineno}: duplicate objective entry")
                    cost[j] = v
                elif rn in dropped_n:
                    continue
                else:
                    i = row_index.get(rn)
                    if i is None:
                        raise FormatError(f"line {lineno}: unknown row {rn!r}")
                    if (i, j) in entries:
                        raise FormatError(f"line {lineno}: duplicate entry ({rn}, {cn})")
                    entries[(i, j)] = v
        elif section in ("RHS", "RANGES"):
            if len(tokens) % 2 == 1:
                set_name, pairs = tokens[0], tokens[1:]
            else:
                set_name, pairs = "", tokens
            if section == "RHS":
                if rhs_set is None:
                    rhs_set = set_name
                elif set_name != rhs_set:
                    raise FormatError(f"line {lineno}: more than one RHS set")
            else:
                if range_set is None:
                    range_set = set_name
                elif set_name != range_set:
                    raise FormatError(f"line {lineno}: more than one RANGES set")
            for k in range(0, len(pairs), 2):
                rn, v = pairs[k], num(pairs[k + 1], lineno)
                if section == "RHS" and rn == obj_row:
                    c0 = -v
                    continue
                if rn in dropped_n or rn == obj_row:
                    continue
                i = row_index.get(rn)
                if i is None:
                    raise FormatError(f"line {lineno}: unknown row {rn!r}")
                (rhs if section == "RHS" else ranges)[i] = v
        elif section == "BOUNDS":
            t = tokens[0].upper()
            if t in ("BV", "LI", "UI", "SC"):
                raise FormatError(f"line {lineno}: integer bound type {t} is not supported")
            no_value = t in ("FR", "MI", "PL")
            if t not in ("UP", "LO", "FX", "FR", "MI", "PL"):
                raise FormatError(f"line {lineno}: unknown bound type {t!r}")
            want = 2 if no_value else 3
            if len(tokens) == want + 1:
                set_name, rest = tokens[1], tokens[2:]
            elif len(tokens) == want:
                set_name, rest = "", tokens[1:]
            else:
                raise FormatError(f"line {lineno}: malformed bound")
            if bound_set is None:
                bound_set = set_name
            elif set_name != bound_set:
                raise FormatError(f"line {lineno}: more than one BOUNDS set")
            j = col(rest[0])
            v = None if no_value else num(rest[1], lineno)
            if t == "UP":
                assert v is not None
                if v < 0 and j not in lb:
                    lb[j] = None
                    neg_up_warn.append(col_names[j])
                ub[j] = v
            elif t == "LO":
                lb[j] = v
            elif t == "FX":
                lb[j] = v
                ub[j] = v
            elif t == "FR":
                lb[j] = None
                ub[j] = None
            elif t == "MI":
                lb[j] = None
            elif t == "PL":
                ub[j] = None
        elif section in ("QUADOBJ", "QSECTION", "QMATRIX"):
            if len(tokens) != 3:
                raise FormatError(f"line {lineno}: expected 'col col value' in {section}")
            a, b2, v = col(tokens[0]), col(tokens[1]), num(tokens[2], lineno)
            key = (a, b2)
            if key in quad:
                raise FormatError(f"line {lineno}: duplicate quadratic entry")
            quad[key] = v
        else:
            raise FormatError(f"line {lineno}: data outside a section")

    if not seen_end:
        raise FormatError("missing ENDATA")
    if obj_row is None:
        raise FormatError("no objective (N) row")

    m, n = len(row_type), len(col_names)
    A = QMatrix.from_triplets(
        (m, n), [k[0] for k in entries], [k[1] for k in entries], list(entries.values())
    )
    rl: list[Any | None] = []
    ru: list[Any | None] = []
    for i, t in enumerate(row_type):
        b = rhs.get(i, ZERO)
        r = ranges.get(i)
        if t == "E":
            lo, hi = b, b
            if r is not None:
                if r > 0:
                    hi = b + r
                elif r < 0:
                    lo = b + r
        elif t == "L":
            lo, hi = (None if r is None else b - abs(r)), b
        else:
            lo, hi = b, (None if r is None else b + abs(r))
        rl.append(lo)
        ru.append(hi)

    P = _quadratic(quad, quad_kind, n)
    sign = sense
    c = [cost.get(j, ZERO) for j in range(n)]
    if sign == -1:  # store as minimisation of the negated objective
        c = [-v for v in c]
        c0 = -c0
        P = QMatrix(P.shape, P.indptr, P.indices, [-v for v in P.data])
    prob = BoxedQP(
        P=P,
        c=c,
        A=A,
        rl=rl,
        ru=ru,
        lb=[lb.get(j, ZERO) for j in range(n)],
        ub=[ub.get(j) for j in range(n)],
        c0=c0,
        name=name,
        objective_sign=sign,
        row_names=row_names,
        col_names=col_names,
        warnings=(
            [f"negative UP bound set the lower bound to -inf for {len(neg_up_warn)} columns"]
            if neg_up_warn
            else []
        ),
    )
    return prob


def _sense(tok: str, lineno: int) -> int:
    t = tok.upper()
    if t in ("MIN", "MINIMIZE", "MINIMISE"):
        return 1
    if t in ("MAX", "MAXIMIZE", "MAXIMISE"):
        return -1
    raise FormatError(f"line {lineno}: unknown objective sense {tok!r}")


def _quadratic(quad: dict[tuple[int, int], Any], kind: str | None, n: int) -> QMatrix:
    rows: list[dict[int, Any]] = [{} for _ in range(n)]
    if kind == "full":
        for (i, j), v in quad.items():
            rows[i][j] = v
        P = QMatrix.from_rows((n, n), rows)
        if not P.is_symmetric():
            raise FormatError("QMATRIX is not symmetric")
        return P
    for (i, j), v in quad.items():
        if i != j and (j, i) in quad:
            raise FormatError("QUADOBJ lists both (i, j) and (j, i); it takes one triangle")
        rows[i][j] = v
        rows[j][i] = v
    return QMatrix.from_rows((n, n), rows)


__all__ = ["FormatError", "Q", "parse_qps", "read_qps"]
