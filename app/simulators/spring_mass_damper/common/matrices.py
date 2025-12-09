import numpy as np

from .parameters import SmdParams


def smd_matrices(params: SmdParams):
    m = params.mass
    k = params.k
    c = params.c
    A = np.array([[0.0, 1.0], [-k / m, -c / m]], dtype=float)
    B = np.array([[0.0], [1.0 / m]], dtype=float)
    C = np.array([[1.0, 0.0]], dtype=float)
    return A, B, C
