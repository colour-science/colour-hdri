#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debevec (1997) Camera Response Function Computation
===================================================

Defines *Debevec (1997)* camera responses computation objects:

-   :func:`g_solve`
-   :func:`camera_response_functions_Debevec1997`

See Also
--------
`Colour - HDRI - Examples: Merge from Low Dynamic Range Files Jupyter Notebook
<https://github.com/colour-science/colour-hdri/\
blob/master/colour_hdri/examples/examples_merge_from_ldr_files.ipynb>`_

References
----------
.. [1]  Debevec, P., & Malik, J. (1997). Recovering High Dynamic Range Radiance
        Maps from Photographs, (August), 1–10. doi:10.1145/258734.258884
"""

from __future__ import division, unicode_literals

import numpy as np

from colour import tstack

from colour_hdri.generation import weighting_function_Debevec1997
from colour_hdri.sampling import samples_Grossberg2003
from colour_hdri.utilities import average_luminance

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['g_solve',
           'camera_response_functions_Debevec1997']


def g_solve(Z, B, l=30, w=weighting_function_Debevec1997, n=256):
    """
    Given a set of pixel values observed for several pixels in several images
    with different exposure times, this function returns the imaging system’s
    response function :math:`g` as well as the log film irradiance values
    :math:`lE` for the observed pixels.

    Parameters
    ----------
    Z : array_like
        Set of pixel values observed for several pixels in several images.
    B : array_like
        Log :math:`\Delta t`, or log shutter speed for images.
    l : numeric, optional
        :math:`\lambda` smoothing term.
    w : callable, optional
        Weighting function :math:`w`.
    n : int, optional
        :math:`n` constant.

    Returns
    -------
    tuple
        Camera response functions :math:`g(z)` and log film irradiance values
        :math:`lE`.
    """

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

    A[k, np.int_(n / 2)] = 1
    k += 1

    for i in np.arange(1, n - 1, 1):
        A[k, i - 1] = l * w[i]
        A[k, i + 0] = l * w[i] * -2
        A[k, i + 1] = l * w[i]
        k += 1

    x = np.squeeze(np.linalg.lstsq(A, b)[0])

    g = x[0:n]
    lE = x[n:x.shape[0]]

    return g, lE


def camera_response_functions_Debevec1997(image_stack,
                                          s=samples_Grossberg2003,
                                          samples=1000,
                                          l=30,
                                          w=weighting_function_Debevec1997,
                                          n=256,
                                          normalise=True):
    """
    Returns the camera response functions for given image stack using
    *Debevec (1997)* method.

    Image channels are sampled with :math:`s` sampling function and the output
    samples are passed to :func:`g_solve`.

    Parameters
    ----------
    image_stack : ImageStack
        Stack of single channel or multi-channel floating point images.
    s : callable, optional
        Sampling function :math:`s`.
    samples : int, optional
        Samples count per images.
    l : numeric, optional
        :math:`\lambda` smoothing term.
    w : callable, optional
        Weighting function :math:`w`.
    n : int, optional
        :math:`n` constant.
    normalise : bool, optional
        Enables the camera response functions normalisation. Uncertain camera
        response functions values resulting from :math:`w` function are
        set to zero.

    Returns
    -------
    ndarray
        Camera response functions :math:`g(z)`.
    """

    samples = s(image_stack.data, samples, n)

    L_l = np.log(average_luminance(image_stack.f_number,
                                   image_stack.exposure_time,
                                   image_stack.iso))

    crfs = np.exp(tstack(np.array([g_solve(samples[..., x], L_l, l, w, n)[0]
                                   for x in range(samples.shape[-1])])))

    if normalise:
        # TODO: Investigate if the normalisation value should account for the
        # percentage of uncertain camera response functions values or be
        # correlated to it and scaled accordingly. As an alternative of setting
        # the uncertain camera response functions values to zero, it would be
        # interesting to explore extrapolation as the camera response functions
        # are essentially smooth. It is important to note that camera sensors
        # are usually acting non linearly when reaching saturation level.
        crfs[w(np.linspace(0, 1, crfs.shape[0])) == 0] = 0
        crfs /= np.max(crfs, axis=0)

    return crfs
