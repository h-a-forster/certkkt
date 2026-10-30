"""Interval estimates used in the report."""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (``(0, 1)`` when n = 0)."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + level / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Clamp the closed-form endpoints at k = 0 and k = n against rounding.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def fmt_share(k: int, n: int) -> str:
    """``k/n = p% [lo, hi]`` with a 95% Wilson interval, in percent."""
    if n == 0:
        return "0/0"
    lo, hi = wilson(k, n)
    pct = (200 * k + n) // (2 * n)  # round half up in integers (float 100*k/n ties vary)
    return f"{k}/{n} = {pct}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (type 7) of a non-empty list."""
    if not values:
        raise ValueError("empty list")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


__all__ = ["fmt_share", "holm", "mcnemar_exact", "quantile", "wilson"]
