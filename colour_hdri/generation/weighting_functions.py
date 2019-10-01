# -*- coding: utf-8 -*-
"""
Weighting Functions
===================

Defines weighting function objects used when generating radiance images:

-   :func:`colour_hdri.normal_distribution_function`
-   :func:`colour_hdri.hat_function`
-   :func:`colour_hdri.weighting_function_Debevec1997`

References
----------
-   :cite:`Debevec1997a` : Debevec, P. E., & Malik, J. (1997). Recovering high
    dynamic range radiance maps from photographs. In Proceedings of the 24th
    annual conference on Computer graphics and interactive techniques -
    SIGGRAPH '97 (pp. 369-378). New York, New York, USA: ACM Press.
    doi:10.1145/258734.258884
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'normal_distribution_function', 'hat_function',
    'weighting_function_Debevec1997'
]


def normal_distribution_function(a, mu=0.5, sigma=0.15):
    """
    Returns given array weighted by a normal distribution function.

    Parameters
    ----------
    a : array_like
        Array to apply the weighting function onto.
    mu : numeric, optional
        Mean or expectation.
    sigma : numeric, optional
        Standard deviation.

    Returns
    -------
    ndarray
        Weighted array.

    Examples
    -------
    >>> normal_distribution_function(np.linspace(0, 1, 10))
    array([ 0.00386592,  0.03470859,  0.18002174,  0.53940751,  0.93371212,
            0.93371212,  0.53940751,  0.18002174,  0.03470859,  0.00386592])
    """

    a = as_float_array(a)

    return np.exp(-np.power(a - mu, 2) / (2 * np.power(sigma, 2)))


def hat_function(a):
    """
    Returns given array weighted by a hat function.

    Parameters
    ----------
    a : array_like
        Array to apply the weighting function onto.

    Returns
    -------
    ndarray
        Weighted array.

    Examples
    -------
    >>> hat_function(np.linspace(0, 1, 10))
    array([ 0.        ,  0.95099207,  0.99913557,  0.99999812,  1.        ,
            1.        ,  0.99999812,  0.99913557,  0.95099207,  0.        ])
    """

    a = as_float_array(a)

    return 1 - (2 * a - 1) ** 12


def weighting_function_Debevec1997(a, domain_l=0.01, domain_h=0.99):
    """
    Returns given array weighted by *Debevec (1997)* function.

    Parameters
    ----------
    a : array_like
        Array to apply the weighting function onto.
    domain_l : numeric, optional
        Domain lowest possible value, values less than ``domain_l`` will be set
        to zero.
    domain_h : numeric, optional
        Domain highest possible value, values greater than ``domain_h`` will be
        set to zero.

    Returns
    -------
    ndarray
        Weighted array.

    References
    ----------
    :cite:`Debevec1997a`

    Examples
    -------
    >>> weighting_function_Debevec1997(np.linspace(0, 1, 10))
    array([ 0.        ,  0.23273657,  0.48849105,  0.74424552,  1.        ,
            1.        ,  0.74424552,  0.48849105,  0.23273657,  0.        ])
    """

    a = as_float_array(a)

    w = np.zeros(a.shape)

    mask = np.where(a <= (domain_l + domain_h) / 2, True, False)
    w[mask] = a[mask] - domain_l
    w[~mask] = domain_h - a[~mask]
    w /= np.max(w)

    w[w < 0] = 0

    return w
