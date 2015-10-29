#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour import dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['highlights_recovery_clip',
           'highlights_recovery_blend']


def highlights_recovery_clip(RGB, whitepoint, threshold=0.99):
    RGB = np.copy(RGB)
    whitepoint = np.asarray(whitepoint)

    RGB[np.any(RGB >= threshold, axis=2)] = whitepoint

    return RGB


def highlights_recovery_blend(RGB, white_level, threshold=0.99):
    M = np.array([[1, 1, 1],
                  [1.7320508, -1.7320508, 0],
                  [-1, -1, 2]])

    clipping_level = white_level * threshold

    Lab = dot_vector(M, RGB)

    Lab_c = dot_vector(M, np.minimum(RGB, clipping_level))

    s = np.sum((Lab * Lab)[..., 1:3], axis=2)
    s_c = np.sum((Lab_c * Lab_c)[..., 1:3], axis=2)

    ratio = np.sqrt(s_c / s)
    ratio[np.logical_or(np.isnan(ratio), np.isinf(ratio))] = 1

    Lab[:, :, 1:3] *= np.rollaxis(ratio[np.newaxis], 0, 3)

    RGB_o = dot_vector(np.linalg.inv(M), Lab)

    return RGB_o
