import random

import numpy as np
import pytest
import scipy.sparse as sp

from certkkt.rational import Q, to_q
from certkkt.sparse import BudgetExceeded, QMatrix, psd_check, solve_any, vstack


def _rand_qmatrix(rng, m, n, density=0.4, lo=-5, hi=5):
    rows = []
    for _ in range(m):
        rows.append(
            {
                j: Q(rng.randint(lo, hi), rng.randint(1, 4))
                for j in range(n)
                if rng.random() < density
            }
        )
    return QMatrix.from_rows((m, n), rows)


def test_matvec_rmatvec_transpose_match_dense():
    rng = random.Random(0)
    A = _rand_qmatrix(rng, 7, 5)
    x = [Q(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(5)]
    y = [Q(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(7)]
    dense = [[A.row_dicts()[i].get(j, Q(0)) for j in range(5)] for i in range(7)]
    assert A.matvec(x) == [sum((dense[i][j] * x[j] for j in range(5)), Q(0)) for i in range(7)]
    assert A.rmatvec(y) == [sum((dense[i][j] * y[i] for i in range(7)), Q(0)) for j in range(5)]
    assert A.transpose().transpose() == A
    assert A.transpose().matvec(y) == A.rmatvec(y)


def test_from_triplets_sums_duplicates_and_drops_zeros():
    A = QMatrix.from_triplets((2, 2), [0, 0, 1, 1], [0, 0, 1, 1], ["0.5", "0.5", 1, -1])
    assert A.nnz == 1 and A.row(0) == ([0], [Q(1)])
    with pytest.raises(ValueError):
        QMatrix.from_triplets((2, 2), [2], [0], [1])


def test_to_scipy_is_correctly_rounded_and_32_bit():
    A = QMatrix.from_triplets((1, 2), [0, 0], [0, 1], ["0.1", "3"])
    S = A.to_scipy()
    assert S.indices.dtype == np.int32
    assert S.toarray()[0, 0] == 0.1
    assert QMatrix.from_scipy(sp.csr_array(np.array([[0.1, 0.0]]))).data == [to_q(0.1)]


def test_vstack():
    a = QMatrix.identity(2)
    b = QMatrix.from_rows((1, 2), [{1: Q(3)}])
    s = vstack([a, b])
    assert s.shape == (3, 2) and s.row(2) == ([1], [Q(3)])


@pytest.mark.parametrize("seed", range(20))
def test_solve_any_square_and_rectangular(seed):
    rng = random.Random(seed)
    m, n = rng.randint(1, 8), rng.randint(1, 8)
    A = _rand_qmatrix(rng, m, n, density=0.5)
    x_true = [Q(rng.randint(-5, 5), rng.randint(1, 3)) for _ in range(n)]
    rhs = A.matvec(x_true)  # consistent by construction
    x = solve_any(A.row_dicts(), rhs, n, col_penalty={0: 1})
    assert x is not None and A.matvec(x) == rhs


def test_solve_any_detects_inconsistency():
    rows = [{0: Q(1), 1: Q(1)}, {0: Q(2), 1: Q(2)}]
    assert solve_any(rows, [Q(1), Q(3)], 2) is None
    x = solve_any(rows, [Q(1), Q(2)], 2)
    assert x is not None and x[0] + x[1] == 1


def test_solve_any_budget():
    rng = random.Random(5)
    A = _rand_qmatrix(rng, 30, 30, density=0.9)
    with pytest.raises(BudgetExceeded):
        solve_any(A.row_dicts(), [Q(1)] * 30, 30, budget=10)


def test_psd_check():
    rng = random.Random(3)
    for _ in range(20):
        k, n = rng.randint(1, 4), rng.randint(2, 6)
        B = [[Q(rng.randint(-3, 3)) for _ in range(n)] for _ in range(k)]
        G = [[sum((B[t][i] * B[t][j] for t in range(k)), Q(0)) for j in range(n)] for i in range(n)]
        M = QMatrix.from_rows((n, n), [{j: G[i][j] for j in range(n) if G[i][j]} for i in range(n)])
        assert psd_check(M).is_psd
        if k < n:  # rank deficient: subtracting a tiny multiple of I breaks PSD
            G2 = [[G[i][j] - (Q(1, 10**9) if i == j else 0) for j in range(n)] for i in range(n)]
            M2 = QMatrix.from_rows(
                (n, n), [{j: G2[i][j] for j in range(n) if G2[i][j]} for i in range(n)]
            )
            assert psd_check(M2).status == "not_psd"
    indef = QMatrix.from_rows((2, 2), [{0: Q(1), 1: Q(2)}, {0: Q(2), 1: Q(1)}])
    assert psd_check(indef).status == "not_psd"
    with pytest.raises(ValueError):
        psd_check(QMatrix.from_rows((2, 2), [{1: Q(1)}, {}]))
