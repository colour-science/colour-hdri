# -*- coding: utf-8 -*-
"""
RGB Colourspace & Transformations
=================================

Defines the following *RGB* colourspace transformations:

-   :func:`colour_hdri.camera_space_to_RGB`
-   :func:`colour_hdri.camera_space_to_sRGB`
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models import RGB_COLOURSPACES
from colour.utilities import matrix_dot, vector_dot

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['camera_space_to_RGB', 'camera_space_to_sRGB']


def camera_space_to_RGB(RGB, M_XYZ_to_camera_space, matrix_RGB_to_XYZ):
    """
    Converts given *RGB* array from *camera space* to given *RGB* colourspace.

    Parameters
    ----------
    RGB : array_like
        Camera space *RGB* colourspace array.
    M_XYZ_to_camera_space : array_like
        Matrix converting from *CIE XYZ* tristimulus values to *camera space*.
    matrix_RGB_to_XYZ : array_like
        Matrix converting from *RGB* colourspace to *CIE XYZ* tristimulus
        values.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Examples
    --------
    >>> RGB = np.array([0.80660, 0.81638, 0.65885])
    >>> M_XYZ_to_camera_space = np.array([
    ...     [0.47160000, 0.06030000, -0.08300000],
    ...     [-0.77980000, 1.54740000, 0.24800000],
    ...     [-0.14960000, 0.19370000, 0.66510000]])
    >>> matrix_RGB_to_XYZ = np.array([
    ...     [0.41238656, 0.35759149, 0.18045049],
    ...     [0.21263682, 0.71518298, 0.07218020],
    ...     [0.01933062, 0.11919716, 0.95037259]])
    >>> camera_space_to_RGB(
    ...     RGB,
    ...     M_XYZ_to_camera_space,
    ...     matrix_RGB_to_XYZ)  # doctest: +ELLIPSIS
    array([ 0.7564180...,  0.8683192...,  0.6044589...])
    """

    M_RGB_camera = matrix_dot(M_XYZ_to_camera_space, matrix_RGB_to_XYZ)

    M_RGB_camera /= np.transpose(np.sum(M_RGB_camera, axis=1)[np.newaxis])

    RGB_f = vector_dot(np.linalg.inv(M_RGB_camera), RGB)

    return RGB_f


def camera_space_to_sRGB(RGB, M_XYZ_to_camera_space):
    """
    Converts given *RGB* array from *camera space* to *sRGB* colourspace.

    Parameters
    ----------
    RGB : array_like
        Camera space *RGB* colourspace array.
    M_XYZ_to_camera_space : array_like
        Matrix converting from *CIE XYZ* tristimulus values to *camera space*.

    Returns
    -------
    ndarray
        *sRGB* colourspace array.

    Examples
    --------
    >>> RGB = np.array([0.80660, 0.81638, 0.65885])
    >>> M_XYZ_to_camera_space = np.array([
    ...     [0.47160000, 0.06030000, -0.08300000],
    ...     [-0.77980000, 1.54740000, 0.24800000],
    ...     [-0.14960000, 0.19370000, 0.66510000]])
    >>> camera_space_to_sRGB(RGB, M_XYZ_to_camera_space)  # doctest: +ELLIPSIS
    array([ 0.7564350...,  0.8683155...,  0.6044706...])
    """

    return camera_space_to_RGB(RGB, M_XYZ_to_camera_space,
                               RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ)
