#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['highlights_recovery_clip']


def highlights_recovery_clip(RGB, whitepoint, threshold=0.95):
    RGB = np.copy(RGB)
    whitepoint = np.asarray(whitepoint)

    RGB[np.any(RGB >= threshold, axis=2)] = whitepoint

    return RGB
