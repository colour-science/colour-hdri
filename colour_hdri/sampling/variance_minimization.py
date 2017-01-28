#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Viriyothai (2009) Variance Minimization Light Probe Sampling
============================================================

Defines *Viriyothai (2009)* variance minimization light probe sampling objects:

-   :func:`light_probe_sampling_variance_minimization_Viriyothai2009`

References
----------
.. [1]  Viriyothai, K., & Debevec, P. (2009). Variance minimization light probe
        sampling. SIGGRAPH’09: Posters, (Egsr), 60558.
        doi:10.1145/1599301.1599393
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour import RGB_COLOURSPACES, RGB_luminance, centroid, warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Light_Specification',
           'luminance_variance',
           'find_regions_variance_minimization_Viriyothai2009',
           'highlight_regions_variance_minimization',
           'light_probe_sampling_variance_minimization_Viriyothai2009']


class Light_Specification(
        namedtuple('Light_Specification', ('uv', 'colour', 'index'))):
    """
    Defines a light probe sampling resulting light specification.

    Parameters
    ----------
    uv : array_like
        :math:`uv` coordinates of the sampled light.
    colour : array_like
        Sampled light colour.
    index : array_like
        Sampled light location index in its original array.
    """


def luminance_variance(a):
    """
    Computes the Luminance variance of given :math:`a` 2-D array.

    Parameters
    ----------
    a : array_like
        :math:`a` 2-D array to compute the Luminance variation.

    Returns
    -------
    numeric
        :math:`a` 2-D array Luminance variance.

    Examples
    --------
    >>> a = np.tile(np.arange(5), (5, 1))
    >>> luminance_variance(a)  # doctest: +ELLIPSIS
    12.2474487...
    """

    x, y = np.mgrid[0:np.shape(a)[0], 0:np.shape(a)[1]]

    x_centroid, y_centroid = centroid(a)

    variance = np.sqrt(np.sum(
        a * ((y - y_centroid) ** 2 + (x - x_centroid) ** 2)))

    return variance


def find_regions_variance_minimization_Viriyothai2009(a, n=4):
    """
    Finds the :math:`2^n` regions using *Viriyothai (2009)* variance
    minimization light probe sampling algorithm on given :math:`a` 2-D array.

    Parameters
    ----------
    a : array_like
        :math:`a` 2-D array to find the regions.
    n : int
        Iterations count, the total regions count is :math:`2^n`.

    Returns
    -------
    list
        Regions with variance minimized.
    """

    a = np.asarray(a)

    regions = [(0, a.shape[0], 0, a.shape[1])]

    for i in range(n):
        sub_regions = []
        for region in regions:
            variance = np.inf
            horizontal = False
            location = -1
            y_min, y_max, x_min, x_max = region

            # Vertical cut on image width.
            for j in range(x_min, x_max):
                variance_c = max(
                    luminance_variance(a[y_min:y_max, x_min:j]),
                    luminance_variance(a[y_min:y_max, j:x_max]))

                if variance_c < variance:
                    variance = variance_c
                    horizontal = False
                    location = j

            # Horizontal cut on image height.
            for j in range(y_min, y_max):
                variance_c = max(
                    luminance_variance(a[y_min:j, x_min:x_max]),
                    luminance_variance(a[j:y_max, x_min:x_max]))

                if variance_c < variance:
                    variance = variance_c
                    horizontal = True
                    location = j

            if horizontal:
                sub_regions.append((y_min, location, x_min, x_max))
                sub_regions.append((location, y_max, x_min, x_max))
            else:
                sub_regions.append((y_min, y_max, x_min, location))
                sub_regions.append((y_min, y_max, location, x_max))

        regions = sub_regions

    return regions


def highlight_regions_variance_minimization(
        a,
        regions,
        highlight_colour=np.array([0, 1, 0])):
    """
    Highlights regions using with variance minimized on given :math:`a`
    3-D array.

    Parameters
    ----------
    a : array_like
        :math:`a` 3-D array to highlight the regions.
    regions : array_like
        Regions with variance minimized.
    highlight_colour : array_like
        Highlight colour.

    Returns
    -------
    ndarray
        :math:`a` 3-D array with highlighted regions.
    """

    a = np.copy(a)

    for region in regions:
        y_min, y_max, x_min, x_max = region

        a[y_min:y_max, x_min] = highlight_colour
        a[y_min:y_max, x_max - 1] = highlight_colour
        a[y_min, x_min:x_max] = highlight_colour
        a[y_max - 1, x_min:x_max] = highlight_colour

    return a


def light_probe_sampling_variance_minimization_Viriyothai2009(
        light_probe,
        lights_count=16,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Sample given light probe to find lights using *Viriyothai (2009)* variance
    minimization light probe sampling algorithm.

    Parameters
    ----------
    light_probe : array_like
        Array to sample for lights.
    lights_count : int
        Amount of lights to generate.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    list
        list of :class:`Light_Specification` lights.
    """

    light_probe = np.asarray(light_probe)

    iterations = np.sqrt(lights_count).astype(np.int_)
    if iterations ** 2 != lights_count:
        warning(
            '{0} lights requested, {1} will be effectively computed!'.format(
                lights_count, iterations ** 2))

    Y = RGB_luminance(
        light_probe, colourspace.primaries, colourspace.whitepoint)
    regions = find_regions_variance_minimization_Viriyothai2009(Y, iterations)

    lights = []
    for region in regions:
        y_min, y_max, x_min, x_max = region
        c = centroid(Y[y_min:y_max, x_min:x_max])
        c = (c + np.array([y_min, x_min]))
        lights.append(
            Light_Specification(
                (c / np.array(Y.shape))[::-1],
                np.sum(np.sum(light_probe[y_min:y_max, x_min:x_max], 0), 0),
                c))

    return lights
