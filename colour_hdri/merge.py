#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour_hdri.exposure import average_luminance
from colour_hdri.weighting_functions import weighting_function_Debevec1997


def merge_to_hdri(image_stack,
                  weighting_function=weighting_function_Debevec1997):
    image_c = None
    weight_c = None
    for image in image_stack:
        if image_c is None:
            image_c = np.zeros(image.pixel_data.shape)
            weight_c = np.zeros(image.pixel_data.shape)

        L = average_luminance(
            image.exposure_data.aperture,
            image.exposure_data.shutter_speed,
            image.exposure_data.iso)

        weight = weighting_function(image.pixel_data)

        image_c += weight * image.pixel_data / L
        weight_c += weight

    if image_c is not None:
        image_c /= weight_c
        image_c[np.isnan(image_c)] = 0

    return image_c
