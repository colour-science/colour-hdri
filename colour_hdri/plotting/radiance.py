#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HDRI / Radiance Image Plotting
==============================

Defines the HDRI / radiance image plotting objects:

-   :func:`radiance_image_strip_plot`
"""

from __future__ import division, unicode_literals

import matplotlib.pyplot
import numpy as np

from colour.plotting import DEFAULT_PLOTTING_ENCODING_CCTF, display

from colour_hdri.utilities.exposure import adjust_exposure

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['radiance_image_strip_plot']


def radiance_image_strip_plot(image,
                              count=5,
                              ev_steps=-2,
                              encoding_cctf=DEFAULT_PLOTTING_ENCODING_CCTF,
                              **kwargs):
    """
    Plots given HDRI / radiance image as strip of images of varying exposure.

    Parameters
    ----------
    image : array_like
         HDRI / radiance image to plot.
    count : int, optional
        Strip images count.
    ev_steps : numeric, optional
        Exposure variation for each image of the strip.
    encoding_cctf : callable, optional
        Encoding colour component transfer function / opto-electronic
        transfer function used for plotting.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`display`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    Figure
        Current figure or None.
    """

    image = np.asarray(image)

    grid = matplotlib.gridspec.GridSpec(1, count)
    grid.update(wspace=0, hspace=0)

    height, width, _channel = image.shape
    for i in range(count):
        ev = i * ev_steps
        axis = matplotlib.pyplot.subplot(grid[i])
        axis.imshow(
            np.clip(encoding_cctf(adjust_exposure(image, ev)), 0, 1))
        axis.text(width * 0.05,
                  height - height * 0.05,
                  'EV {0}'.format(ev),
                  color=(1, 1, 1))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect('equal')

    return display(**kwargs)
