#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour_hdri.exposure import average_luminance
from colour_hdri.weighting_functions import weighting_function_Debevec1997

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['radiance_image']


def radiance_image(image_stack,
                   weighting_function=weighting_function_Debevec1997,
                   weighting_average=False):
    image_c = None
    weight_c = None
    for image in image_stack:
        if image_c is None:
            image_c = np.zeros(image.data.shape)
            weight_c = np.zeros(image.data.shape)

        L = average_luminance(
            image.metadata.f_number,
            image.metadata.exposure_time,
            image.metadata.iso)

        if weighting_average and image.data.ndim == 3:
            weight = weighting_function(np.average(image.data, axis=-1))
            weight = np.rollaxis(weight[np.newaxis], 0, 3)
        else:
            weight = weighting_function(image.data)

        image_c += weight * image.data / L
        weight_c += weight

    if image_c is not None:
        image_c /= weight_c
        image_c[np.isnan(image_c)] = 0

    return image_c
