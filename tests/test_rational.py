import math
import random
from fractions import Fraction

import pytest

from certkkt.rational import (
    Q,
    as_double,
    q_float_down,
    q_float_up,
    simplest_between,
    snap,
    sqrt_down,
    sqrt_up,
    to_q,
)


def test_decimal_strings_are_exact():
    assert to_q("0.1") == Q(1, 10)
    assert to_q("-.100000e+01") == -1
    assert to_q("10.") == 10
    assert to_q("1.5D3") == 1500
    assert to_q("+3.5e-2") == Q(7, 200)


@pytest.mark.parametrize(
    "bad", ["", ".", "1e", "e5", "1/3", "0x10", "inf", "nan", "1_000", "1e999999"]
)
def test_bad_decimals_rejected(bad):
    with pytest.raises(ValueError):
        to_q(bad)


def test_floats_convert_to_their_dyadic_value():
    assert to_q(0.1) == Fraction(0.1)
    assert to_q(0.1) != Q(1, 10)
    with pytest.raises(ValueError):
        to_q(float("nan"))
    with pytest.raises(TypeError):
        to_q(True)


def test_as_double_rounds_to_nearest():
    assert as_double(Q(1, 10)) == to_q(0.1)
    assert as_double(Q(1, 3)) == to_q(1 / 3)


def test_directed_rounding_brackets_value():
    rng = random.Random(1)
    for _ in range(2000):
        v = Q(rng.randint(-(10**30), 10**30), rng.randint(1, 10**25))
        lo, hi = q_float_down(v), q_float_up(v)
        assert to_q(lo) <= v <= to_q(hi)
        assert hi == lo or math.nextafter(lo, math.inf) == hi
    assert q_float_up(Q(1, 10)) == 0.1  # the double 0.1 lies just above 1/10
    assert q_float_down(Q(1, 10)) == math.nextafter(0.1, 0.0)
    assert q_float_up(Q(10**400)) == math.inf
    assert q_float_down(Q(10**400)) == 1.7976931348623157e308
    assert q_float_down(Q(-(10**400))) == -math.inf


def test_sqrt_bounds():
    for v in [Q(2), Q(1, 3), Q(10**-30), Q(123456789, 7), Q(10**40 + 1)]:
        lo, hi = sqrt_down(v), sqrt_up(v)
        assert lo * lo <= v <= hi * hi
        assert hi - lo <= hi * Q(1, 2**55)
    assert sqrt_up(Q(9, 4)) == Q(3, 2) == sqrt_down(Q(9, 4))
    assert sqrt_up(Q(0)) == 0


def test_simplest_between_matches_brute_force():
    rng = random.Random(2)
    for _ in range(300):
        a = Fraction(rng.randint(-200, 200), rng.randint(1, 60))
        b = a + Fraction(rng.randint(0, 50), rng.randint(1, 400))
        got = simplest_between(Q(a.numerator, a.denominator), Q(b.numerator, b.denominator))
        best = None
        for den in range(1, 500):
            lo = math.ceil(a * den)
            hi = math.floor(b * den)
            if lo <= hi:
                num = min(range(lo, hi + 1), key=abs)
                best = Fraction(num, den)
                break
        assert best is not None
        assert Fraction(int(got.numerator), int(got.denominator)) == best


def test_snap_recovers_simple_fractions():
    assert snap(1 / 3, 1e-12) == Q(1, 3)
    assert snap(0.1 + 0.2, 1e-12) == Q(3, 10)
    assert snap(2.5e-17, 1e-9) == 0
    assert snap(math.pi, 0) == to_q(math.pi)
