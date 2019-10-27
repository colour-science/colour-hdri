# -*- coding: utf-8 -*-
"""
HDRI / Radiance Image Plotting
==============================

Defines the HDRI / radiance image plotting objects:

-   :func:`colour_hdri.plotting.plot_radiance_image_strip`
"""

from __future__ import division, unicode_literals

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from colour.plotting import COLOUR_STYLE_CONSTANTS, override_style, render
from colour.utilities import as_float_array

from colour_hdri.utilities.exposure import adjust_exposure

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['plot_radiance_image_strip']


@override_style()
def plot_radiance_image_strip(
        image,
        count=5,
        ev_steps=-2,
        cctf_encoding=COLOUR_STYLE_CONSTANTS.colour.colourspace.cctf_encoding,
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
    cctf_encoding : callable, optional
        Encoding colour component transfer function / opto-electronic
        transfer function used for plotting.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.display`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    tuple
        Current figure and axes.
    """

    image = as_float_array(image)

    grid = matplotlib.gridspec.GridSpec(1, count)
    grid.update(wspace=0, hspace=0)

    height, width, _channel = image.shape
    for i in range(count):
        ev = i * ev_steps
        axis = plt.subplot(grid[i])
        axis.imshow(np.clip(cctf_encoding(adjust_exposure(image, ev)), 0, 1))
        axis.text(
            width * 0.05,
            height - height * 0.05,
            'EV {0}'.format(ev),
            color=(1, 1, 1))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect('equal')

    return render(**kwargs)
