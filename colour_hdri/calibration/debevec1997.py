#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour import tstack

from colour_hdri.generation import weighting_function_Debevec1997
from colour_hdri.sampling import samples_Grossberg2009
from colour_hdri.utilities import average_luminance

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['g_solve',
           'camera_response_function_Debevec1997']


def g_solve(Z, B, l, w=weighting_function_Debevec1997, n=256):
    Z = np.asarray(Z).astype(int)
    B = np.asarray(B)
    l = np.asarray(l)

    Z_x, Z_y = Z.shape

    A = np.zeros((Z_x * Z_y + n - 1, n + Z_x))
    b = np.zeros((A.shape[0], 1))
    w = w(np.linspace(0, 1, n))

    k = 0
    for i in np.arange(Z_x):
        for j in np.arange(Z_y):
            Z_c = Z[i, j]
            w_ij = w[Z_c]
            A[k, Z_c] = w_ij
            A[k, n + i] = -w_ij
            b[k] = w_ij * B[j]
            k += 1

    A[k, n / 2] = 1
    k += 1

    for i in np.arange(1, n - 1, 1):
        A[k, i - 1] = l * w[i]
        A[k, i + 0] = l * w[i] * -2
        A[k, i + 1] = l * w[i]
        k += 1

    x = np.squeeze(np.linalg.lstsq(A, b)[0])

    g = x[0:n]
    l_E = x[n:x.shape[0]]

    return g, l_E


def camera_response_function_Debevec1997(image_stack,
                                         samples=1000,
                                         l=30,
                                         n=256,
                                         w=weighting_function_Debevec1997,
                                         normalise=True):
    samples = samples_Grossberg2009(image_stack.data, samples, n)

    L_l = np.log(average_luminance(image_stack.f_number,
                                   image_stack.exposure_time,
                                   image_stack.iso))

    crfs = np.exp(tstack(np.array([g_solve(samples[..., x], L_l, l, w, n)[0]
                                   for x in range(samples.shape[-1])])))

    if normalise:
        crfs[w(np.linspace(0, 1, crfs.shape[0])) == 0] = 0
        crfs /= np.max(crfs, axis=0)

    return crfs
