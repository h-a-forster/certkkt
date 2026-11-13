# certkkt study: quick profile

244 units. Shares are `k/n = p% [95% Wilson interval]`.

Outcomes: certified 243, no_solution 1

Repair time budget exhausted: cblib 12, mm 15 units.

## Certification at solver defaults

As-is: the solver's floats, converted exactly; every exact relative residual
(primal, dual, gap) at most eps. Repaired: an exactly feasible primal-dual
pair (level optimal or bounded) with exact relative gap at most eps.

As-is at own tol: the same test at the solver's documented default tolerance
(in certkkt's measure, which is not the solver's stopping criterion).

| suite | solver | n | claims optimal | as-is 1e-4 | as-is 1e-6 | as-is 1e-8 | as-is at own tol | repaired 1e-6 | repaired 1e-8 | exact optimum |
|---|---|---|---|---|---|---|---|---|---|---|
| cblib | clarabel | 4 | 3/4 = 75% [30, 95] | 4/4 = 100% [51, 100] | 3/4 = 75% [30, 95] | 3/4 = 75% [30, 95] | 3/4 = 75% [30, 95] at 1e-8 | 0/4 = 0% [0, 49] | 0/4 = 0% [0, 49] | 0/4 = 0% [0, 49] |
| cblib | ecos | 4 | 3/4 = 75% [30, 95] | 3/4 = 75% [30, 95] | 3/4 = 75% [30, 95] | 3/4 = 75% [30, 95] | 3/4 = 75% [30, 95] at 1e-8 | 0/4 = 0% [0, 49] | 0/4 = 0% [0, 49] | 0/4 = 0% [0, 49] |
| cblib | scs | 4 | 4/4 = 100% [51, 100] | 3/4 = 75% [30, 95] | 1/4 = 25% [5, 70] | 1/4 = 25% [5, 70] | 3/4 = 75% [30, 95] at 1e-4 | 0/4 = 0% [0, 49] | 0/4 = 0% [0, 49] | 0/4 = 0% [0, 49] |
| mm | clarabel | 40 | 40/40 = 100% [91, 100] | 40/40 = 100% [91, 100] | 37/40 = 93% [80, 97] | 30/40 = 75% [60, 86] | 30/40 = 75% [60, 86] at 1e-8 | 30/40 = 75% [60, 86] | 30/40 = 75% [60, 86] | 23/40 = 58% [42, 71] |
| mm | highs | 40 | 39/40 = 98% [87, 100] | 35/40 = 88% [74, 95] | 23/40 = 58% [42, 71] | 5/40 = 13% [5, 26] | 14/40 = 35% [22, 50] at 1e-7 | 31/40 = 78% [62, 88] | 31/40 = 78% [62, 88] | 29/40 = 73% [57, 84] |
| mm | osqp | 40 | 33/40 = 83% [68, 91] | 7/40 = 18% [9, 32] | 3/40 = 8% [3, 20] | 0/40 = 0% [0, 9] | 21/40 = 53% [37, 67] at 1e-3 | 19/40 = 48% [33, 63] | 19/40 = 48% [33, 63] | 19/40 = 48% [33, 63] |
| mm | scs | 40 | 40/40 = 100% [91, 100] | 30/40 = 75% [60, 86] | 16/40 = 40% [26, 55] | 7/40 = 18% [9, 32] | 30/40 = 75% [60, 86] at 1e-4 | 23/40 = 58% [42, 71] | 23/40 = 58% [42, 71] | 22/40 = 55% [40, 69] |
| portfolio | clarabel | 6 | 6/6 = 100% [61, 100] | 6/6 = 100% [61, 100] | 6/6 = 100% [61, 100] | 1/6 = 17% [3, 56] | 1/6 = 17% [3, 56] at 1e-8 | 6/6 = 100% [61, 100] | 3/6 = 50% [19, 81] | 0/6 = 0% [0, 39] |
| portfolio | ecos | 6 | 6/6 = 100% [61, 100] | 6/6 = 100% [61, 100] | 6/6 = 100% [61, 100] | 0/6 = 0% [0, 39] | 0/6 = 0% [0, 39] at 1e-8 | 6/6 = 100% [61, 100] | 5/6 = 83% [44, 97] | 0/6 = 0% [0, 39] |
| portfolio | scs | 6 | 6/6 = 100% [61, 100] | 6/6 = 100% [61, 100] | 3/6 = 50% [19, 81] | 0/6 = 0% [0, 39] | 6/6 = 100% [61, 100] at 1e-4 | 1/6 = 17% [3, 56] | 0/6 = 0% [0, 39] | 0/6 = 0% [0, 39] |

Exact optimum paired by instance (mm, 40 problems): 29 reached from at least one solver, 19 from all 4. Exact McNemar test, Holm-adjusted over the 6 pairs.

| a | b | only a | only b | p | Holm p |
|---|---|---|---|---|---|
| clarabel | highs | 0 | 6 | 0.031 | 0.125 |
| clarabel | osqp | 4 | 0 | 0.125 | 0.375 |
| clarabel | scs | 1 | 0 | 1.000 | 1.000 |
| highs | osqp | 10 | 0 | 0.002 | 0.012 |
| highs | scs | 7 | 0 | 0.016 | 0.078 |
| osqp | scs | 0 | 3 | 0.250 | 0.500 |

Float64 and exact evaluation of the as-is residuals disagree at eps = 1e-4 on 0 of 243 units with both.
Float64 and exact evaluation of the as-is residuals disagree at eps = 1e-6 on 0 of 243 units with both.
Float64 and exact evaluation of the as-is residuals disagree at eps = 1e-8 on 0 of 243 units with both.
Largest |log10(float64 / exact)| of max_rel: 0.15; largest absolute difference: 2e-11. Units with an exact max_rel within a factor of 2 of eps: 19 of 1e-4, 12 of 1e-6, 18 of 1e-8.

Certificate level as returned (as-is), over units with a solution: residuals 216, primal_feasible 27.
Best certificate after repair, cblib: residuals 12.
Best certificate after repair, mm: optimal 93, residuals 38, bounded 16, dual_feasible 12, none 1.
Best certificate after repair, portfolio: bounded 72.

## Requested tolerance versus certified accuracy

Median [quartiles] of log10 of the exact as-is relative residual (largest of
primal, dual, gap) and of the exact relative gap alone, over units with a
solution; `<= tol` is the share whose exact residual meets the requested
tolerance under certkkt's relative measure (solvers scale differently).

| suite | solver | tol | n | log10 max_rel | log10 rel_gap | max_rel <= tol | repaired at tol |
|---|---|---|---|---|---|---|---|
| cblib | clarabel | default | 4 | -8.2 [-8.3, -7.2] | -10.7 [-11.0, -9.9] | - | - |
| cblib | ecos | default | 4 | -9.4 [-10.3, -7.1] | -12.0 [-12.7, -10.7] | - | - |
| cblib | scs | default | 4 | -4.1 [-5.4, -3.3] | -9.1 [-9.7, -8.7] | - | - |
| mm | clarabel | default | 40 | -9.0 [-10.0, -7.9] | -9.6 [-10.0, -8.9] | - | - |
| mm | highs | default | 40 | -6.5 [-7.5, -5.5] | -7.3 [-8.2, -5.6] | - | - |
| mm | osqp | default | 40 | -3.0 [-3.8, -1.4] | -3.4 [-4.3, -3.0] | - | - |
| mm | scs | default | 40 | -5.1 [-7.6, -3.8] | -7.2 [-8.6, -5.3] | - | - |
| portfolio | clarabel | default | 6 | -7.6 [-7.9, -7.4] | -10.2 [-10.6, -9.8] | - | - |
| portfolio | clarabel | 1e-4 | 6 | -3.6 [-3.8, -3.4] | -5.8 [-6.1, -5.5] | 0/6 = 0% [0, 39] | 2/6 = 33% [10, 70] |
| portfolio | clarabel | 1e-6 | 6 | -6.1 [-6.2, -5.7] | -8.6 [-8.7, -8.4] | 4/6 = 67% [30, 90] | 4/6 = 67% [30, 90] |
| portfolio | clarabel | 1e-8 | 6 | -7.6 [-7.9, -7.4] | -10.2 [-10.6, -9.8] | 1/6 = 17% [3, 56] | 3/6 = 50% [19, 81] |
| portfolio | ecos | default | 6 | -7.5 [-7.6, -7.3] | -10.0 [-10.2, -10.0] | - | - |
| portfolio | ecos | 1e-4 | 6 | -3.6 [-3.9, -3.4] | -6.2 [-6.5, -6.0] | 2/6 = 33% [10, 70] | 6/6 = 100% [61, 100] |
| portfolio | ecos | 1e-6 | 6 | -5.4 [-5.7, -5.3] | -8.1 [-8.3, -8.0] | 1/6 = 17% [3, 56] | 6/6 = 100% [61, 100] |
| portfolio | ecos | 1e-8 | 6 | -7.5 [-7.6, -7.3] | -10.0 [-10.2, -10.0] | 0/6 = 0% [0, 39] | 5/6 = 83% [44, 97] |
| portfolio | scs | default | 6 | -6.1 [-6.6, -4.7] | -8.4 [-9.3, -7.1] | - | - |
| portfolio | scs | 1e-4 | 6 | -6.1 [-6.6, -4.7] | -8.4 [-9.3, -7.1] | 6/6 = 100% [61, 100] | 4/6 = 67% [30, 90] |
| portfolio | scs | 1e-6 | 6 | -7.1 [-9.7, -6.6] | -9.9 [-10.9, -8.8] | 5/6 = 83% [44, 97] | 3/6 = 50% [19, 81] |
| portfolio | scs | 1e-8 | 6 | -9.2 [-10.1, -8.6] | -11.3 [-11.5, -10.9] | 5/6 = 83% [44, 97] | 4/6 = 67% [30, 90] |

