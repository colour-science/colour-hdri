#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clipped Highlights Recovery
===========================

Defines the clipped highlights recovery objects:

-   :func:`highlights_recovery_blend`
-   :func:`highlights_recovery_LCHab`

See Also
--------
`Colour - HDRI - Examples: Merge from Raw Files Jupyter Notebook
<https://github.com/colour-science/colour-hdri/\
blob/master/colour_hdri/examples/examples_merge_from_raw_files.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour import (
    LCHab_to_Lab,
    Lab_to_LCHab,
    Lab_to_XYZ,
    RGB_to_XYZ,
    XYZ_to_Lab,
    XYZ_to_RGB,
    dot_vector,
    sRGB_COLOURSPACE,
    tsplit,
    tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['highlights_recovery_blend',
           'highlights_recovery_LCHab']


def highlights_recovery_blend(RGB, multipliers, threshold=0.99):
    """
    Performs highlights recovery using *Coffin (1997)* method from *dcraw*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    multipliers : array_like
        Normalised camera white level or white balance multipliers.
    threshold : numeric, optional
        Threshold for highlights selection.

    Returns
    -------
    ndarray
         Highlights recovered *RGB* colourspace array.

    References
    ----------
    .. [1]  Coffin, D. (2015). dcraw. Retrieved from
            https://www.cybercom.net/~dcoffin/dcraw/
    """

    M = np.array([[1.0000000, 1.0000000, 1.0000000],
                  [1.7320508, -1.7320508, 0.0000000],
                  [-1.0000000, -1.0000000, 2.0000000]])

    clipping_level = np.min(multipliers) * threshold

    Lab = dot_vector(M, RGB)

    Lab_c = dot_vector(M, np.minimum(RGB, clipping_level))

    s = np.sum((Lab * Lab)[..., 1:3], axis=2)
    s_c = np.sum((Lab_c * Lab_c)[..., 1:3], axis=2)

    ratio = np.sqrt(s_c / s)
    ratio[np.logical_or(np.isnan(ratio), np.isinf(ratio))] = 1

    Lab[:, :, 1:3] *= np.rollaxis(ratio[np.newaxis], 0, 3)

    RGB_o = dot_vector(np.linalg.inv(M), Lab)

    return RGB_o


def highlights_recovery_LCHab(
        RGB, threshold=None, RGB_colourspace=sRGB_COLOURSPACE):
    """
    Performs highlights recovery in *CIE LCHab* colourspace.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    threshold : numeric, optional
        Threshold for highlights selection, automatically computed
        if not given.
    RGB_colourspace : RGB_Colourspace, optional
        Working *RGB* colourspace to perform the *CIE LCHab* to and from.

    Returns
    -------
    ndarray
         Highlights recovered *RGB* colourspace array.
    """

    L, _C, H = tsplit(Lab_to_LCHab(XYZ_to_Lab(RGB_to_XYZ(
        RGB,
        RGB_colourspace.whitepoint,
        RGB_colourspace.whitepoint,
        RGB_colourspace.RGB_to_XYZ_matrix),
        RGB_colourspace.whitepoint)))
    _L_c, C_c, _H_c = tsplit(Lab_to_LCHab(XYZ_to_Lab(RGB_to_XYZ(
        np.clip(RGB, 0, threshold),
        RGB_colourspace.whitepoint,
        RGB_colourspace.whitepoint,
        RGB_colourspace.RGB_to_XYZ_matrix),
        RGB_colourspace.whitepoint)))

    return XYZ_to_RGB(Lab_to_XYZ(LCHab_to_Lab(
        tstack((L, C_c, H))), RGB_colourspace.whitepoint),
        RGB_colourspace.whitepoint,
        RGB_colourspace.whitepoint,
        RGB_colourspace.XYZ_to_RGB_matrix)
