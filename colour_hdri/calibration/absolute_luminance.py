#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Absolute Luminance Calibration - Lagarde (2016)
===============================================

Defines *Lagarde (2016)* panoramic images absolute *Luminance* calibration
objects:

-   :func:`absolute_luminance_calibration_Lagarde2016`
-   :func:`upper_hemisphere_illuminance_Lagarde2016`
-   :func:`upper_hemisphere_illuminance_weights_Lagarde2016`

References
----------
.. [1]  Lagarde, S., Lachambre, S., & Jover, C. (2016). An Artist-Friendly
        Workflow for Panoramic HDRI. Retrieved from
        http://blog.selfshadow.com/publications/s2016-shading-course/\
unity/s2016_pbs_unity_hdri_notes.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour import RGB_COLOURSPACES, RGB_luminance

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['upper_hemisphere_illuminance_Lagarde2016',
           'upper_hemisphere_illuminance_weights_Lagarde2016',
           'absolute_luminance_calibration_Lagarde2016']


def upper_hemisphere_illuminance_Lagarde2016(
        RGB,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Computes upper hemisphere illuminance :math:`E_v` of given RGB panoramic
    image.

    Parameters
    ----------
    RGB : array_like
        *RGB* panoramic image array.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    numeric
        Upper hemisphere illuminance :math:`E_v`.

    Examples
    --------
    >>> RGB = np.ones((16, 32, 3))
    >>> upper_hemisphere_illuminance_Lagarde2016(RGB)  # doctest: +ELLIPSIS
    2.9344691...
    """

    RGB = np.asarray(RGB)

    height, width, channels = RGB.shape

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)

    theta = np.linspace(0, 1, height) * np.pi

    theta_cos = np.cos(theta)[..., np.newaxis]
    theta_sin = np.sin(theta)[..., np.newaxis]

    E_v = np.sum(np.where(theta_cos > 0, L * theta_cos * theta_sin, 0))

    E_v *= 2 * np.pi ** 2 / (width * height)

    return E_v


def upper_hemisphere_illuminance_weights_Lagarde2016(height, width):
    """
    Computes upper hemisphere illuminance weights for use with applications
    unable to perform the computation directly, i.e. *Adobe Photoshop*.

    Parameters
    ----------
    height : int
        Output array height.
    width : int
        Output array width.

    Returns
    -------
    ndarray
        Upper hemisphere illuminance weights.

    Examples
    --------
    >>> upper_hemisphere_illuminance_weights_Lagarde2016(  # doctest: +ELLIPSIS
    ...    16, 1)
    array([[ 0...        ],
           [ 4.0143297...],
           [ 7.3345454...],
           [ 9.3865515...],
           [ 9.8155376...],
           [ 8.5473281...],
           [ 5.8012079...],
           [ 2.0520061...],
           [ 0...        ],
           [ 0...        ],
           [ 0...        ],
           [ 0...        ],
           [ 0...        ],
           [ 0...        ],
           [ 0...        ],
           [ 0...        ]])
    """

    w = np.zeros((height, width))

    theta = (np.linspace(0, 1, height) * np.pi)
    theta = np.tile(theta[..., np.newaxis], (1, width))

    theta_cos = np.cos(theta)
    theta_sin = np.sin(theta)

    w[theta_cos > 0] = (theta_cos[theta_cos > 0] * theta_sin[theta_cos > 0] *
                        2 * np.pi ** 2)

    return w


def absolute_luminance_calibration_Lagarde2016(
        RGB,
        measured_illuminance,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs absolute *Luminance* calibration of given *RGB* panoramic image
    using *Lagarde (2016)* method.

    Parameters
    ----------
    RGB : array_like
        *RGB* panoramic image to calibrate.
    measured_illuminance : numeric
        Measured illuminance :math:`E_v`.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
         Absolute *Luminance* calibrated *RGB* panoramic image.

    Examples
    --------
    >>> RGB = np.ones((4, 8, 3))
    >>> absolute_luminance_calibration_Lagarde2016(  # doctest: +ELLIPSIS
    ...     RGB, 500)
    array([[[ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...]],
    <BLANKLINE>
           [[ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...]],
    <BLANKLINE>
           [[ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...]],
    <BLANKLINE>
           [[ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...],
            [ 233.9912506...,  233.9912506...,  233.9912506...]]])
    """

    RGB = np.asarray(RGB)

    E_v = upper_hemisphere_illuminance_Lagarde2016(RGB, colourspace)

    return RGB / E_v * measured_illuminance
