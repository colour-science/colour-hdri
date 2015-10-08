#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour_hdri.weighting_functions import weighting_function_Debevec1997


def g_solve(Z, B, l, w=weighting_function_Debevec1997):
    Z = np.asarray(Z).astype(int)
    B = np.asarray(B)
    l = np.asarray(l)

    Z_x, Z_y = Z.shape

    n = 256
    A = np.zeros((Z_x * Z_y + n + 1, n + Z_x))
    b = np.zeros((A.shape[0], 1))
    w = w(np.linspace(0, 1, n))

    k = 0
    for i in np.arange(Z_x):
        for j in np.arange(Z_y):
            w_ij = w[Z[i, j]]
            A[k, Z[i, j]] = w_ij
            A[k, n + i] = -w_ij
            b[k] = w_ij * B[j]
            k += 1

    A[k, 128] = 1
    k += 1

    for i in np.arange(n - 2):
        A[k, i] = l * w[i + 1]
        A[k, i + 1] = -2 * l * w[i + 1]
        A[k, i + 2] = l * w[i + 1]
        k += 1

    x = np.linalg.lstsq(A, b)[0]

    g = x[0:n]
    lE = x[n:x.shape[0]]

    return g, lE
