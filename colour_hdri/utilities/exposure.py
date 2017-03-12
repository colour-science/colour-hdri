#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Exposure Value Computation
================================

Defines image exposure value computation objects:

-   :func:`exposure_value`
-   :func:`adjust_exposure`
-   :func:`average_luminance`
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['exposure_value',
           'adjust_exposure',
           'average_luminance']


def exposure_value(f_number, exposure_time, iso):
    """
    Computes the exposure value from given image *FNumber*, *Exposure Time* and
    *ISO* values.

    Parameters
    ----------
    f_number : array_like
        Image *FNumber*.
    exposure_time : array_like
        Image *Exposure Time*.
    iso : array_like
        Image *ISO*.

    Returns
    -------
    ndarray
        Image exposure value.

    Examples
    --------
    >>> exposure_value(8, 1, 100)
    6.0
    """

    N = np.asarray(f_number)
    t = np.asarray(exposure_time)
    S = np.asarray(iso)

    EV = np.log2(N ** 2) + np.log2(1 / t) - np.log2(100 / S)

    return EV


def adjust_exposure(a, EV):
    """
    Adjusts given array exposure using given :math:`EV` exposure value.

    Parameters
    ----------
    a : array_like
        Array to adjust the exposure.
    EV : numeric
        Exposure adjustment value.

    Returns
    -------
    ndarray
        Exposure adjusted array.

    Examples
    --------
    >>> adjust_exposure(np.array([0.25, 0.5, 0.75, 1]), 1)
    array([ 0.5,  1. ,  1.5,  2. ])
    """

    a = np.asarray(a)

    return a * pow(2, EV)


def average_luminance(f_number, exposure_time, iso, k=12.5):
    """
    Computes the average luminance from given image *FNumber*, *Exposure Time*
    and *ISO* values.

    Parameters
    ----------
    f_number : array_like
        Image *FNumber*.
    exposure_time : array_like
        Image *Exposure Time*.
    iso : array_like
        Image *ISO*.
    k : numeric, optional
        Reflected light calibration constant :math:`K`.

    Returns
    -------
    ndarray
        Image average luminance.

    References
    ----------
    .. [1]  Wikipedia. (n.d.). EV as a measure of luminance and illuminance.
            Retrieved November 14, 2015,
            from https://en.wikipedia.org/wiki/\
Exposure_value#EV_as_a_measure_of_luminance_and_illuminance

    Examples
    --------
    >>> average_luminance(8, 1, 100)
    0.125
    """

    N = np.asarray(f_number)
    t = np.asarray(exposure_time)
    S = np.asarray(iso)

    L = (S * t) / (k * N ** 2)

    return L
