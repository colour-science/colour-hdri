#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tonemapping Operators Plotting
==============================

Defines the tonemapping operators plotting objects:

-   :func:`radiance_image_strip_plot`
"""

from __future__ import division, unicode_literals

import matplotlib
import matplotlib.pyplot
import matplotlib.ticker
import numpy as np
import pylab

from colour.plotting import (
    DEFAULT_PLOTTING_ENCODING_CCTF,
    boundaries,
    canvas,
    display,
    decorate)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['tonemapping_operator_image_plot']


def tonemapping_operator_image_plot(
        image,
        luminance_function,
        log_scale=False,
        encoding_cctf=DEFAULT_PLOTTING_ENCODING_CCTF,
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
    encoding_cctf : callable, optional
        Encoding colour component transfer function / opto-electronic
        transfer function used for plotting.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`boundaries`, :func:`canvas`, :func:`decorate`,
        :func:`display`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    bool
        Definition success.
    """

    shape = image.shape
    limits = [0, 1, 0, 1]

    image = np.clip(encoding_cctf(image), 0, 1)
    pylab.imshow(image,
                 aspect=shape[0] / shape[1],
                 extent=limits,
                 interpolation='nearest')

    pylab.plot(np.linspace(0, 1, len(luminance_function)),
               luminance_function,
               color='red')

    settings = {
        'figure_size': (8, 8),
        'x_label': 'Input Luminance',
        'y_label': 'Output Luminance',
        'x_ticker': True,
        'y_ticker': True,
        'grid': True,
        'x_tighten': True,
        'y_tighten': True,
        'limits': limits}
    settings.update(kwargs)

    if log_scale:
        settings.update({
            'x_label': '$log_2$ Input Luminance',
            'x_ticker_locator': matplotlib.ticker.AutoMinorLocator(0.5)})
        matplotlib.pyplot.gca().set_xscale('log', basex=2)
        matplotlib.pyplot.gca().xaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter())

    canvas(**settings)
    decorate(**settings)
    boundaries(**settings)

    return display(**settings)
