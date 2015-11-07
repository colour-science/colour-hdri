#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour import RGB_COLOURSPACES, dot_matrix, dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['camera_space_to_RGB',
           'camera_space_to_sRGB']


def camera_space_to_RGB(RGB, XYZ_to_camera_matrix, RGB_to_XYZ_matrix):
    M_RGB_camera = dot_matrix(XYZ_to_camera_matrix, RGB_to_XYZ_matrix)

    M_RGB_camera /= np.transpose(np.sum(M_RGB_camera, axis=1)[np.newaxis])

    RGB_f = dot_vector(np.linalg.inv(M_RGB_camera), RGB)

    return RGB_f


def camera_space_to_sRGB(RGB, XYZ_to_camera_matrix):
    return camera_space_to_RGB(
        RGB, XYZ_to_camera_matrix, RGB_COLOURSPACES['sRGB'].RGB_to_XYZ_matrix)
