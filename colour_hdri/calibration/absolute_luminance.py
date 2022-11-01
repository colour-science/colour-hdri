"""
Absolute Luminance Calibration - Lagarde (2016)
===============================================

Defines the *Lagarde (2016)* panoramic images absolute *Luminance* calibration
objects:

-   :func:`colour_hdri.absolute_luminance_calibration_Lagarde2016`
-   :func:`colour_hdri.upper_hemisphere_illuminance_Lagarde2016`
-   :func:`colour_hdri.upper_hemisphere_illuminance_weights_Lagarde2016`

References
----------
-   :cite:`Lagarde2016b` : Lagarde, Sebastien, Lachambre, S., & Jover, C.
    (2016). An Artist-Friendly Workflow for Panoramic HDRI.
    http://blog.selfshadow.com/publications/s2016-shading-course/unity/\
s2016_pbs_unity_hdri_notes.pdf
"""

from __future__ import annotations

import numpy as np

from colour import RGB_COLOURSPACES, RGB_Colourspace, RGB_luminance
from colour.hints import ArrayLike, Floating, Integer, NDArray
from colour.utilities import as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "upper_hemisphere_illuminance_Lagarde2016",
    "upper_hemisphere_illuminance_weights_Lagarde2016",
    "absolute_luminance_calibration_Lagarde2016",
]


def upper_hemisphere_illuminance_Lagarde2016(
    RGB: ArrayLike, colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"]
) -> Floating:
    """
    Compute upper hemisphere illuminance :math:`E_v` of given RGB panoramic
    image.

    Parameters
    ----------
    RGB
        *RGB* panoramic image array.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.floating`
        Upper hemisphere illuminance :math:`E_v`.

    References
    ----------
    :cite:`Lagarde2016b`

    Examples
    --------
    >>> RGB = np.ones((16, 32, 3))
    >>> upper_hemisphere_illuminance_Lagarde2016(RGB)  # doctest: +ELLIPSIS
    2.9344691...
    """

    RGB = as_float_array(RGB)

    height, width, _channels = RGB.shape

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)

    theta = np.linspace(0, 1, height) * np.pi

    theta_cos = np.cos(theta)[..., None]
    theta_sin = np.sin(theta)[..., None]

    E_v = np.sum(np.where(theta_cos > 0, L * theta_cos * theta_sin, 0))

    E_v *= 2 * np.pi**2 / (width * height)

    return E_v


def upper_hemisphere_illuminance_weights_Lagarde2016(
    height: Integer, width: Integer
) -> NDArray:
    """
    Compute upper hemisphere illuminance weights for use with applications
    unable to perform the computation directly, i.e. *Adobe Photoshop*.

    Parameters
    ----------
    height
        Output array height.
    width
        Output array width.

    Returns
    -------
    :class:`numpy.ndarray`
        Upper hemisphere illuminance weights.

    References
    ----------
    :cite:`Lagarde2016b`

    Examples
    --------
    >>> upper_hemisphere_illuminance_weights_Lagarde2016(  # doctest: +ELLIPSIS
    ...     16, 1
    ... )
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

    theta = np.linspace(0, 1, height) * np.pi
    theta = np.tile(theta[..., None], (1, width))

    theta_cos = np.cos(theta)
    theta_sin = np.sin(theta)

    w[theta_cos > 0] = (
        theta_cos[theta_cos > 0] * theta_sin[theta_cos > 0] * 2 * np.pi**2
    )

    return w


def absolute_luminance_calibration_Lagarde2016(
    RGB: ArrayLike,
    measured_illuminance: Floating,
    colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"],
) -> NDArray:
    """
    Perform absolute *Luminance* calibration of given *RGB* panoramic image
    using *Lagarde (2016)* method.

    Parameters
    ----------
    RGB
        *RGB* panoramic image to calibrate.
    measured_illuminance
        Measured illuminance :math:`E_v`.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
         Absolute *Luminance* calibrated *RGB* panoramic image.

    Examples
    --------
    >>> RGB = np.ones((4, 8, 3))
    >>> absolute_luminance_calibration_Lagarde2016(  # doctest: +ELLIPSIS
    ...     RGB, 500
    ... )
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

    RGB = as_float_array(RGB)

    E_v = upper_hemisphere_illuminance_Lagarde2016(RGB, colourspace)

    return RGB / E_v * measured_illuminance
