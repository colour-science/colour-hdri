# -*- coding: utf-8 -*-
"""
Image Exposure Value Computation
================================

Defines image exposure value computation objects:

-   :func:`colour_hdri.average_luminance`
-   :func:`colour_hdri.average_illuminance`
-   :func:`colour_hdri.exposure_value`
-   :func:`colour_hdri.adjust_exposure`

References
----------
-   :cite:`Wikipediabj` : Wikipedia. (n.d.). EV as a measure of luminance and
    illuminance. Retrieved November 14, 2015, from https://en.wikipedia.org/\
wiki/Exposure_value#EV_as_a_measure_of_luminance_and_illuminance
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'average_luminance', 'average_illuminance', 'exposure_value',
    'adjust_exposure'
]


def average_luminance(f_number, exposure_time, iso, k=12.5):
    """
    Computes the average luminance in :math:`cd\\cdot m^{-2}` from given
    image *F-Number* :math:`N`, *Exposure Time* :math:`t`, *ISO* speed
    :math:`S` and *reflected light calibration constant* :math:`k`.

    Parameters
    ----------
    f_number : array_like
        Image *F-Number*  :math:`N`.
    exposure_time : array_like
        Image *Exposure Time* :math:`t`.
    iso : array_like
        Image *ISO* :math:`S`.
    k : numeric, optional
        Reflected light calibration constant :math:`k`.

    Returns
    -------
    ndarray
        Image average luminance in :math:`cd\\cdot m^{-2}`.

    References
    ----------
    :cite:`Wikipediabj`

    Examples
    --------
    >>> average_luminance(8, 1, 100)
    8.0
    """

    N = as_float_array(f_number)
    t = as_float_array(exposure_time)
    S = as_float_array(iso)

    L = N ** 2 / t / S * k

    return L


def average_illuminance(f_number, exposure_time, iso, c=250):
    """
    Computes the average illuminance in :math:`Lux` from given
    image *F-Number* :math:`N`, *Exposure Time* :math:`t`, *ISO* speed
    :math:`S` and *incident light calibration constant* :math:`c`.

    Parameters
    ----------
    f_number : array_like
        Image *F-Number*  :math:`N`.
    exposure_time : array_like
        Image *Exposure Time* :math:`t`.
    iso : array_like
        Image *ISO* :math:`S`.
    c : numeric, optional
        Incident light calibration constant :math:`c`.

    Returns
    -------
    ndarray
        Image average illuminance in :math:`Lux`.

    References
    ----------
    :cite:`Wikipediabj`

    Examples
    --------
    >>> average_illuminance(8, 1, 100)
    160.0
    """

    N = as_float_array(f_number)
    t = as_float_array(exposure_time)
    S = as_float_array(iso)

    E = N ** 2 / t / S * c

    return E


def exposure_value(f_number, exposure_time, iso, k=12.5):
    """
    Computes the average illuminance in :math:`Lux` from given
    image *F-Number* :math:`N`, *Exposure Time* :math:`t` and *ISO* speed
    :math:`S` and *reflected light calibration constant* :math:`k`.

    Parameters
    ----------
    f_number : array_like
        Image *F-Number*  :math:`N`.
    exposure_time : array_like
        Image *Exposure Time* :math:`t`.
    iso : array_like
        Image *ISO* :math:`S`.
    k : numeric, optional
        Reflected light calibration constant :math:`k`.

    Returns
    -------
    ndarray
        Image exposure value.

    Examples
    --------
    >>> exposure_value(8, 1, 100)
    6.0
    """

    L = average_luminance(f_number, exposure_time, iso, k)
    S = as_float_array(iso)

    EV = np.log2(L * S / k)

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

    a = as_float_array(a)

    return a * pow(2, EV)
