# -*- coding: utf-8 -*-
"""
Tonemapping Operators Plotting
==============================

Defines the tonemapping operators plotting objects:

-   :func:`colour_hdri.plotting.plot_tonemapping_operator_image`
"""

from __future__ import division, unicode_literals

import matplotlib
import matplotlib.ticker
import numpy as np

from colour.plotting import (COLOUR_STYLE_CONSTANTS, artist, override_style,
                             render)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['plot_tonemapping_operator_image']


@override_style()
def plot_tonemapping_operator_image(
        image,
        luminance_function,
        log_scale=False,
        cctf_encoding=COLOUR_STYLE_CONSTANTS.colour.colourspace.cctf_encoding,
        **kwargs):
    """
    Plots given tonemapped image with superimposed luminance mapping function.

    Parameters
    ----------
    image : array_like
         Tonemapped image to plot.
    luminance_function : callable
        Luminance mapping function.
    log_scale : bool, optional
        Use a log scale for plotting the luminance mapping function.
    cctf_encoding : callable, optional
        Encoding colour component transfer function / opto-electronic
        transfer function used for plotting.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    tuple
        Current figure and axes.
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    shape = image.shape
    bounding_box = [0, 1, 0, 1]

    image = np.clip(cctf_encoding(image), 0, 1)
    axes.imshow(
        image,
        aspect=shape[0] / shape[1],
        extent=bounding_box,
        interpolation='nearest')

    axes.plot(
        np.linspace(0, 1, len(luminance_function)),
        luminance_function,
        color='red')

    settings = {
        'axes': axes,
        'bounding_box': bounding_box,
        'x_ticker': True,
        'y_ticker': True,
        'x_label': 'Input Luminance',
        'y_label': 'Output Luminance',
    }
    settings.update(kwargs)

    if log_scale:
        settings.update({
            'x_label': '$log_2$ Input Luminance',
            'x_ticker_locator': matplotlib.ticker.AutoMinorLocator(0.5)
        })
        matplotlib.pyplot.gca().set_xscale('log', basex=2)
        matplotlib.pyplot.gca().xaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter())

    return render(**settings)
