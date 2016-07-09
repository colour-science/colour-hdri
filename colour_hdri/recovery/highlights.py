#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clipped Highlights Recovery
===========================

Defines the clipped highlights recovery objects:

-   :func:`highlights_recovery_blend`

See Also
--------
`Colour - HDRI - Examples: Merge from Raw Files IPython Notebook
<https://github.com/colour-science/colour-hdri/\
blob/master/colour_hdri/examples/examples_merge_from_raw_files.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour import dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['highlights_recovery_blend']


def highlights_recovery_blend(RGB, multipliers, threshold=0.99):
    """
    Performs highlights recovery using Coffin (1997) method from *dcraw*.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    multipliers : array_like
        Normalised camera white level or white balance multipliers.
    threshold : numeric, optional
        Threshold for the highlights selection.

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
