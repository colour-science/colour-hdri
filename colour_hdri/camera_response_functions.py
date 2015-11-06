#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour import tsplit, tstack

from colour_hdri.exposure import average_luminance
from colour_hdri.weighting_functions import weighting_function_Debevec1997

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['samples_Grossberg2009',
           'g_solve',
           'camera_response_function_Debevec1997']


def samples_Grossberg2009(image_stack, samples=1000, n=256):
    image_stack = np.asarray(image_stack)

    if image_stack.ndim == 3:
        channels_c = 1
    else:
        channels_c = image_stack.shape[-2]

    cdf_i = []
    for image in tsplit(image_stack):
        histograms = tstack(
            [np.histogram(image[..., c], n, range=(0, 1))[0]
             for c in np.arange(channels_c)])
        cdf = np.cumsum(histograms, axis=0)
        cdf_i.append(cdf.astype(float) / np.max(cdf, axis=0))

    samples_cdf_i = np.zeros((samples, len(cdf_i), channels_c))
    samples_u = np.linspace(0, 1, samples)
    for i in np.arange(samples):
        for j in np.arange(channels_c):
            for k, cdf in enumerate(cdf_i):
                samples_cdf_i[i, k, j] = np.argmin(np.abs(cdf[:, j] -
                                                          samples_u[i]))

    return samples_cdf_i


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
