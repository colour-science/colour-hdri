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

from colour.plotting import DEFAULT_PLOTTING_OECF

from colour_hdri.utilities.exposure import adjust_exposure

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['radiance_image_strip_plot']


def radiance_image_strip_plot(image,
                              count=5,
                              ev_steps=-2,
                              OECF=DEFAULT_PLOTTING_OECF):
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
    OECF : callable, optional
        OECF / opto-electronic conversion function used for plotting.

    Returns
    -------
    bool
        Definition success.
    """

    image = np.asarray(image)

    grid = matplotlib.gridspec.GridSpec(1, count)
    grid.update(wspace=0, hspace=0)

    height, width, _channel = image.shape
    for i in range(count):
        ev = i * ev_steps
        axis = matplotlib.pyplot.subplot(grid[i])
        axis.imshow(
            np.clip(OECF(adjust_exposure(image, ev)), 0, 1))
        axis.text(width * 0.05,
                  height - height * 0.05,
                  'EV {0}'.format(ev),
                  color=(1, 1, 1))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect('equal')

    matplotlib.pyplot.show()

    return True
