import gzip

import pytest

from certkkt import Q, certify, load_instance
from certkkt.cbf import parse_cbf
from certkkt.problem import boxed_duals_to_conic, conic_duals_to_boxed
from certkkt.qps import FormatError, parse_qps

QPS = """\
* small QP exercising most sections
NAME          TINY
ROWS
 N  obj
 G  r1
 L  r2
 E  r3
COLUMNS
    x1        obj       1.5          r1        1.0
    x1        r2        0.1
    x2        obj       -2           r1        1.
    x2        r3        1.0
    x3        r2        -1.0         r3        1.0
RHS
    rhs       obj       -0.5
    rhs       r1        1.0          r2        4.0
    rhs       r3        2.0
RANGES
    rng       r3        -1.0         r2        2.5
BOUNDS
 UP bnd       x1        4.0
 MI bnd       x2
 UP bnd       x3        -1.0
QUADOBJ
    x1        x1        2.0
    x1        x2        0.3
    x2        x2        1.0
ENDATA
"""


def test_qps_sections_are_read_exactly():
    bx = parse_qps(QPS)
    assert bx.name == "TINY" and (bx.m, bx.n) == (3, 3)
    assert bx.c == [Q(3, 2), -2, 0] and bx.c0 == Q(1, 2)
    assert bx.A.row_dicts()[1] == {0: Q(1, 10), 2: -1}  # 0.1 is exactly 1/10
    assert bx.rl == [1, Q(3, 2), 1] and bx.ru == [None, 4, 2]  # L range and E range < 0
    assert bx.lb == [0, None, None] and bx.ub == [4, None, -1]  # negative UP frees x3
    assert bx.warnings
    assert bx.P.row_dicts() == [{0: 2, 1: Q(3, 10)}, {0: Q(3, 10), 1: 1}, {}]


def test_maximisation_is_negated():
    text = QPS.replace("NAME          TINY", "NAME          TINY\nOBJSENSE\n    MAX")
    bx = parse_qps(text)
    assert bx.objective_sign == -1
    assert bx.c[0] == Q(-3, 2) and bx.c0 == Q(-1, 2)
    assert bx.P.row_dicts()[0][0] == -2


def test_fixed_format_names_may_contain_spaces():
    # Fixed-format MPS fields sit in columns 2-3, 5-12, 15-22, 25-36, 40-47 and 50-61, so a
    # name may contain spaces (QFORPLAN has rows such as "DEDO3 1R").
    def data(*f):
        return "    " + "".join(f"{v:<{w}}" for v, w in zip(f, (10, 10, 15, 10, 12), strict=False))

    text = "\n".join(
        [
            "NAME          SPACED",
            "ROWS",
            " N  COST",
            " G  LIM 1",
            " E  MY EQN",
            "COLUMNS",
            data("X 1", "COST", "1.0", "LIM 1", "1.0"),
            data("X 1", "MY EQN", "1.0"),
            data("Y 1", "COST", "2.0", "MY EQN", "-1.0"),
            "RHS",
            data("RHS", "LIM 1", "1.0", "MY EQN", "0.5"),
            "BOUNDS",
            " UP BND       Y 1       4.0",
            "QUADOBJ",
            data("X 1", "X 1", "2.0"),
            "ENDATA",
        ]
    )
    bx = parse_qps(text)
    assert bx.row_names == ["LIM 1", "MY EQN"] and bx.col_names == ["X 1", "Y 1"]
    assert bx.A.row_dicts() == [{0: 1}, {0: 1, 1: -1}] and bx.c == [1, 2]
    assert bx.rl == [1, Q(1, 2)] and bx.ub == [None, 4] and bx.P.row_dicts()[0] == {0: 2}


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda t: t.replace("ENDATA\n", ""), "ENDATA"),
        (lambda t: t.replace(" MI bnd       x2", " BV bnd       x2"), "integer"),
        (
            lambda t: t.replace(
                "    x1        x2        0.3",
                "    x1        x2        0.3\n    x2        x1        0.3",
            ),
            "one triangle",
        ),
        (lambda t: t.replace("-0.5", "0.5x"), "bad number"),
        (lambda t: t.replace(" L  r2", " L  r1"), "duplicate row"),
    ],
)
def test_malformed_qps_fails_loudly(edit, message):
    with pytest.raises(FormatError, match=message):
        parse_qps(edit(QPS))


def test_boxed_to_conic_and_dual_maps_round_trip():
    bx = parse_qps(QPS)
    from certkkt.instance import Instance

    inst = Instance.from_boxed(bx)
    p = inst.problem
    assert [c.kind for c in p.cones] == ["nonneg"]  # the ranged E row is two-sided
    # x feasible for the boxed problem gives a slack in K for the conic one.
    x = [Q(1), Q(3), Q(-3, 2)]
    assert certify(p, x, None).primal_member
    y_row, y_col = [0.5, -1.0, 2.0], [0.0, 0.0, -0.25]
    z = boxed_duals_to_conic(inst.origins, y_row, y_col)
    back_row, back_col = conic_duals_to_boxed(inst.origins, z, bx.m, bx.n)
    assert back_row == [Q(0.5), Q(-1), Q(2)] and back_col == [0, 0, Q(-0.25)]


CBF = """\
VER
3
OBJSENSE
MIN
VAR
4 2
L+ 1
F 3
CON
5 2
L= 1
QR 4
OBJACOORD
1
0 1
ACOORD
6
0 1 1
0 2 1
1 0 1
2 3 1
3 1 1
4 2 1
BCOORD
1
0 -1
"""


def test_cbf_rotated_cone_becomes_exact_soc():
    p = parse_cbf(CBF, name="rq")
    assert [(c.kind, c.dim) for c in p.cones] == [("zero", 1), ("soc", 4), ("nonneg", 1)]
    # Rotated cone (x0, x3, x1, x2): 2 x0 x3 >= x1^2 + x2^2.  Take x1 = x2 = 1/2,
    # x0 = 1, x3 = 1/4: 2 * 1 * 1/4 = 1/2 = 1/4 + 1/4, on the boundary.
    x = [Q(1), Q(1, 2), Q(1, 2), Q(1, 4)]
    cert = certify(p, x, None)
    assert cert.primal_member and cert.primal_residual == 0
    x_bad = [Q(1), Q(1, 2), Q(1, 2), Q(1, 4) - Q(1, 10**12)]
    assert not certify(p, x_bad, None).primal_member


@pytest.mark.parametrize("bad", ["PSDVAR\n1\n2\n", "INT\n1\n0\n", "POWCONES\n0 0\n"])
def test_cbf_unsupported_features_fail_loudly(bad):
    with pytest.raises(FormatError):
        parse_cbf(CBF + bad)


def test_load_instance_reads_gzip(tmp_path):
    f = tmp_path / "tiny.qps.gz"
    f.write_bytes(gzip.compress(QPS.encode()))
    inst = load_instance(f)
    assert inst.boxed is not None and inst.problem.n == 3
    with pytest.raises(ValueError, match="unknown problem file type"):
        load_instance(tmp_path / "x.lp")
