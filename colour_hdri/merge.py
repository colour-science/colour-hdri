#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour_hdri.exposure import average_luminance
from colour_hdri.weighting_functions import weighting_function_Debevec1997


def merge_to_hdri(images,
                  weighting_function=weighting_function_Debevec1997):
    image_c = None
    weight_c = None
    for _path, image, exif_data in images:
        if image_c is None:
            image_c = np.zeros(image.shape)
            weight_c = np.zeros(image.shape)

        L = average_luminance(
            exif_data.aperture, exif_data.shutter_speed, exif_data.iso)

        weight = weighting_function(image)

        image_c += weight * image / L
        weight_c += weight

    if image_c is not None:
        image_c /= weight_c
        image_c[np.isnan(image_c)] = 0

    return image_c
