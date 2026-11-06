"""Do HiGHS's tolerance options change its output on the smallest Maros-Meszaros QPs?

Solves each QP with the adapter's tolerance mapping and with its feasibility part only,
and prints every pair that differs in status or solution; then counts the solutions that
differ across the default, 1e-4 and 1e-8. Needs `certkkt fetch mm`.

    uv run python scripts/highs_tolerances.py [count]   # default 40, about a minute
"""

import sys
from pathlib import Path

import numpy as np

from certkkt.datasets import mm_members
from certkkt.solvers import get_adapter
from certkkt.study import load_unit_instance

FEASIBILITY = ("primal_feasibility_tolerance", "dual_feasibility_tolerance")


def same(r0, r1) -> bool:
    if r0.status != r1.status or (r0.x is None) != (r1.x is None):
        return False
    if (r0.z is None) != (r1.z is None):
        return False
    return (r0.x is None or np.array_equal(r0.x, r1.x)) and (
        r0.z is None or np.array_equal(r0.z, r1.z)
    )


def main() -> None:
    root = Path("data")
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    names = [m[0] for m in sorted(mm_members(root), key=lambda m: m[3])[:count]]
    a = get_adapter("highs")
    differ = moved = 0
    for name in names:
        inst = load_unit_instance("mm", name, root)
        base = a.solve(inst)
        moved += not all(same(base, a.solve(inst, a.tolerance_settings(t))) for t in (1e-4, 1e-8))
        for tol in (1e-1, 1e-2, 1e-4, 1e-8):
            full = a.tolerance_settings(tol)
            r1 = a.solve(inst, full)
            r0 = a.solve(inst, {k: full[k] for k in FEASIBILITY})
            if not same(r0, r1):
                differ += 1
                print(name, tol, r0.status, r1.status, r0.objective, r1.objective)
            elif r1.x is None:
                print(name, tol, f"no solution either way ({r1.status})")
    print(f"{differ} of {4 * len(names)} pairs differ")
    print(f"{moved} of {len(names)} solutions differ across the default, 1e-4 and 1e-8")


if __name__ == "__main__":
    main()
