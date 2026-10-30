"""Exact rational numbers: backend selection, conversions and directed rounding.

All exact arithmetic in certkkt goes through :data:`Q`, which is ``gmpy2.mpq`` when gmpy2
is importable and ``fractions.Fraction`` otherwise. Set ``CERTKKT_RATIONAL=fraction`` to
force the pure-Python backend (the test suite runs a subset under both).

Conversions never round: a binary float converts to the dyadic rational it denotes, and a
decimal string such as ``"0.1"`` converts to ``1/10``. Rounding happens only in the
``*_up`` / ``*_down`` helpers, which return floats that bound the exact value from the
stated side.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Sequence
from fractions import Fraction
from functools import lru_cache
from numbers import Rational
from typing import Any

import numpy as np

_FORCE = os.environ.get("CERTKKT_RATIONAL", "").strip().lower()
if _FORCE not in ("", "gmpy2", "fraction"):
    raise ImportError(f"CERTKKT_RATIONAL must be 'gmpy2' or 'fraction', got {_FORCE!r}")

Q: Any
if _FORCE == "fraction":
    Q = Fraction
    BACKEND = "fraction"
else:
    try:
        from gmpy2 import mpq as _mpq
    except ImportError:  # pragma: no cover - depends on the environment
        if _FORCE == "gmpy2":
            raise
        Q = Fraction
        BACKEND = "fraction"
    else:
        Q = _mpq
        BACKEND = "gmpy2"

ZERO = Q(0)
_QTYPE = type(ZERO)
ONE = Q(1)


def to_q(value: object) -> Any:
    """Convert an int, float, decimal string or rational to an exact rational.

    Floats convert exactly (no decimal round trip). Non-finite floats raise ValueError.
    Strings are parsed as exact decimals, so ``to_q("0.1") == Q(1, 10)``.
    """
    if type(value) is _QTYPE:
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a number here")
    if isinstance(value, float | np.floating):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError(f"cannot convert non-finite float {f!r} to a rational")
        return Q(f) if BACKEND == "gmpy2" else Fraction(f)
    if isinstance(value, int | np.integer):
        return Q(int(value))
    if isinstance(value, str):
        return _parse_decimal(value)
    if isinstance(value, Fraction | Rational):
        return Q(value.numerator, value.denominator)
    if type(value).__name__ == "mpq":  # gmpy2.mpq when the Fraction backend is forced
        return Q(int(value.numerator), int(value.denominator))  # type: ignore[attr-defined]
    raise TypeError(f"cannot convert {type(value).__name__} to a rational")


_DECIMAL = re.compile(r"([+-]?)(\d*)(?:\.(\d*))?(?:[eEdD]([+-]?\d+))?")


@lru_cache(maxsize=65536)
def _parse_decimal(text: str) -> Any:
    """Exact value of a decimal literal such as ``-.1e+01``, ``10.`` or ``1.5D3``."""
    s = text.strip()
    mt = _DECIMAL.fullmatch(s)
    if mt is None or not (mt.group(2) or mt.group(3)):
        raise ValueError(f"not a finite decimal number: {text!r}")
    sign, whole, frac, exp = mt.groups()
    frac = frac or ""
    mant = int((whole or "0") + frac)
    if exp and len(exp.lstrip("+-")) > 5:
        raise ValueError(f"decimal exponent out of range: {text!r}")
    e = (int(exp) if exp else 0) - len(frac)
    if sign == "-":
        mant = -mant
    return Q(mant * 10**e) if e >= 0 else Q(mant, 10**-e)


def as_double(value: Any) -> Any:
    """The rational value of the double nearest to ``value`` (round to nearest even)."""
    return to_q(_nearest(value))


def vec(values: Iterable[object]) -> list[Any]:
    """Convert an iterable (for example a float ndarray) to a list of exact rationals."""
    return [to_q(v) for v in values]


def q_float_down(value: Any) -> float:
    """Largest float that is <= value (correctly handles ties and overflow to -inf)."""
    f = _nearest(value)
    too_big = Q(f) > value if math.isfinite(f) else f > 0
    return math.nextafter(f, -math.inf) if too_big else f


def q_float_up(value: Any) -> float:
    """Smallest float that is >= value."""
    f = _nearest(value)
    too_small = Q(f) < value if math.isfinite(f) else f < 0
    return math.nextafter(f, math.inf) if too_small else f


def _nearest(value: Any) -> float:
    """A float near value: exact integer division gives a correctly rounded result."""
    try:
        return int(value.numerator) / int(value.denominator)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def sqrt_up(value: Any, bits: int = 120) -> Any:
    """A rational upper bound on sqrt(value) with relative error at most about 2**-(bits/2).

    Exact (returns sqrt itself) when value is the square of a rational.
    """
    if value < 0:
        raise ValueError("sqrt of a negative number")
    if value == 0:
        return ZERO
    p, q = int(value.numerator), int(value.denominator)
    # sqrt(p/q) = sqrt(p q) / q; scale by 4**k so that p q 4**k has at least `bits` bits.
    root = math.isqrt(p * q)
    if root * root == p * q:
        return Q(root, q)
    k = max(0, (bits - (p * q).bit_length()) // 2 + 1)
    scaled = p * q << (2 * k)
    root = math.isqrt(scaled)
    if root * root != scaled:
        root += 1
    return Q(root, q << k)


def sqrt_down(value: Any, bits: int = 120) -> Any:
    """A rational lower bound on sqrt(value), exact for squares of rationals."""
    if value < 0:
        raise ValueError("sqrt of a negative number")
    if value == 0:
        return ZERO
    p, q = int(value.numerator), int(value.denominator)
    root = math.isqrt(p * q)
    if root * root == p * q:
        return Q(root, q)
    k = max(0, (bits - (p * q).bit_length()) // 2 + 1)
    return Q(math.isqrt(p * q << (2 * k)), q << k)


def simplest_between(lo: Any, hi: Any) -> Any:
    """The rational with the smallest denominator (then numerator) in the closed [lo, hi].

    Continued-fraction (Stern-Brocot) construction on integer pairs; lo <= hi required.
    """
    if lo > hi:
        raise ValueError("empty interval")
    if lo <= 0 <= hi:
        return ZERO
    neg = hi < 0
    if neg:
        lo, hi = -hi, -lo
    a, b = int(lo.numerator), int(lo.denominator)  # lo = a/b
    c, d = int(hi.numerator), int(hi.denominator)  # hi = c/d, 0 < a/b <= c/d
    p0, q0, p1, q1 = 0, 1, 1, 0  # convergents h_{k-2}, h_{k-1}
    while True:
        fl, rem = divmod(a, b)
        if rem == 0:  # integer lower endpoint
            t = fl
            break
        if c // d > fl:  # an integer lies in (lo, hi]
            t = fl + 1
            break
        p0, q0, p1, q1 = p1, q1, fl * p1 + p0, fl * q1 + q0
        # (lo, hi) <- (1 / (hi - fl), 1 / (lo - fl))
        a, b, c, d = d, c - fl * d, b, rem
    out = Q(t * p1 + p0, t * q1 + q0)
    return -out if neg else out


def snap(value: float, rel_tol: float) -> Any:
    """Simplest rational within ``rel_tol * max(1, |value|)`` of a finite float."""
    x = to_q(value)
    if rel_tol <= 0:
        return x
    tol = to_q(rel_tol) * max(ONE, abs(x))
    return simplest_between(x - tol, x + tol)


def max_abs(values: Sequence[Any]) -> Any:
    """Infinity norm of a sequence of rationals (0 for an empty one)."""
    best = ZERO
    for v in values:
        a = abs(v)
        if a > best:
            best = a
    return best


def dot(a: Sequence[Any], b: Sequence[Any]) -> Any:
    """Exact inner product."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch {len(a)} != {len(b)}")
    total = ZERO
    for u, v in zip(a, b, strict=True):
        if u and v:
            total += u * v
    return total
