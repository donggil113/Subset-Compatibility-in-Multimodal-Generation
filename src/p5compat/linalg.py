"""Tiny dense linear algebra on nested lists (stdlib only).

The FIRST RUN environment has no numpy, so the 3-variable Gaussian
algebra is done here. Matrices are lists of rows; vectors are lists.
Only small symmetric positive-definite systems are expected.
"""

from __future__ import annotations

import math
from typing import List, Sequence

Vector = List[float]
Matrix = List[List[float]]


def submatrix(a: Sequence[Sequence[float]], rows: Sequence[int], cols: Sequence[int]) -> Matrix:
    return [[float(a[i][j]) for j in cols] for i in rows]


def subvector(v: Sequence[float], idx: Sequence[int]) -> Vector:
    return [float(v[i]) for i in idx]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    n, k, m = len(a), len(b), len(b[0])
    if any(len(row) != k for row in a):
        raise ValueError("inner dimensions do not match")
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> Vector:
    return [sum(row[j] * v[j] for j in range(len(v))) for row in a]


def transpose(a: Sequence[Sequence[float]]) -> Matrix:
    return [list(col) for col in zip(*a)]


def cholesky(a: Sequence[Sequence[float]]) -> Matrix:
    """Lower-triangular L with L L^T = a. Raises ValueError if not PD."""
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not positive definite")
                low[i][j] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return low


def _forward(low: Matrix, b: Sequence[float]) -> Vector:
    n = len(low)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(low[i][k] * y[k] for k in range(i))) / low[i][i]
    return y


def _backward_t(low: Matrix, y: Sequence[float]) -> Vector:
    n = len(low)
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
    return x


def spd_solve(a: Sequence[Sequence[float]], b: Sequence[float]) -> Vector:
    low = cholesky(a)
    return _backward_t(low, _forward(low, b))


def spd_inverse(a: Sequence[Sequence[float]]) -> Matrix:
    n = len(a)
    low = cholesky(a)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(_backward_t(low, _forward(low, e)))
    return transpose(cols)


def spd_logdet(a: Sequence[Sequence[float]]) -> float:
    low = cholesky(a)
    return 2.0 * sum(math.log(low[i][i]) for i in range(len(low)))
