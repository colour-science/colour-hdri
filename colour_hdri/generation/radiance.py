#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HDRI / Radiance Image Generation
================================

Defines HDRI / radiance image generation objects:

-   :func:`image_stack_to_radiance_image`

See Also
--------
`Colour - HDRI - Examples Jupyter Notebooks
<https://github.com/colour-science/colour-hdri/\
blob/master/colour_hdri/examples>`_

References
----------
.. [1]  Banterle, F., Artusi, A., Debattista, K., & Chalmers, A. (2011). 2.1.1
        Generating HDR Content by Combining Multiple Exposures. Advanced High
        Dynamic Range Imaging. A K Peters/CRC Press. ISBN:978-1568817194
"""

from __future__ import division, unicode_literals

import numpy as np

from colour import tsplit, tstack

from colour_hdri.generation import weighting_function_Debevec1997
from colour_hdri.utilities import average_luminance

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['image_stack_to_radiance_image']


def image_stack_to_radiance_image(
        image_stack,
        weighting_function=weighting_function_Debevec1997,
        weighting_average=False,
        camera_response_functions=None):
    """
    Generates a HDRI / radiance image from given image stack.

    Parameters
    ----------
    image_stack : ImageStack
        Stack of single channel or multi-channel floating point images. The
        stack is assumed to be representing linear values except if
        ``camera_response_functions`` argument is provided.
    weighting_function : callable, optional
        Weighting function :math:`w`.
    weighting_average : bool, optional
         Enables weighting function :math:`w` computation on channels average
         instead of on a per channel basis.
    camera_response_functions : array_like, optional
        Camera response functions :math:`g(z)` of the imaging system / camera
        if the stack is representing non linear values.

    Returns
    -------
    ndarray
        Radiance image.
    """

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

        image_data = image.data
        if camera_response_functions is not None:
            samples = np.linspace(0, 1, camera_response_functions.shape[0])

            R, G, B = tsplit(image.data)
            R = np.interp(R, samples, camera_response_functions[..., 0])
            G = np.interp(G, samples, camera_response_functions[..., 1])
            B = np.interp(B, samples, camera_response_functions[..., 2])
            image_data = tstack((R, G, B))

        image_c += weight * image_data / L
        weight_c += weight

    if image_c is not None:
        image_c /= weight_c
        image_c[np.isnan(image_c)] = 0

    return image_c
