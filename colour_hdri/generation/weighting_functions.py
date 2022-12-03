"""
Weighting Functions
===================

Defines the weighting function objects used when generating HDRIs:

-   :func:`colour_hdri.normal_distribution_function`
-   :func:`colour_hdri.hat_function`
-   :func:`colour_hdri.weighting_function_Debevec1997`

References
----------
-   :cite:`Debevec1997a` : Debevec, P. E., & Malik, J. (1997). Recovering high
    dynamic range radiance maps from photographs. Proceedings of the 24th
    Annual Conference on Computer Graphics and Interactive Techniques -
    SIGGRAPH "97, August, 369-378. doi:10.1145/258734.258884
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, Floating, NDArray

from colour.utilities import as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "normal_distribution_function",
    "hat_function",
    "weighting_function_Debevec1997",
]


def normal_distribution_function(
    a: ArrayLike, mu: Floating = 0.5, sigma: Floating = 0.15
) -> NDArray:
    """
    Return given array weighted by a normal distribution function.

    Parameters
    ----------
    a
        Array to apply the weighting function onto.
    mu
        Mean or expectation.
    sigma
        Standard deviation.

    Returns
    -------
    :class:`numpy.ndarray`
        Weighted array.

    Examples
    --------
    >>> normal_distribution_function(np.linspace(0, 1, 10))
    array([ 0.00386592,  0.03470859,  0.18002174,  0.53940751,  0.93371212,
            0.93371212,  0.53940751,  0.18002174,  0.03470859,  0.00386592])
    """

    a = as_float_array(a)

    return np.exp(-np.power(a - mu, 2) / (2 * np.power(sigma, 2)))


def hat_function(a: ArrayLike) -> NDArray:
    """
    Return given array weighted by a hat function.

    Parameters
    ----------
    a
        Array to apply the weighting function onto.

    Returns
    -------
    :class:`numpy.ndarray`
        Weighted array.

    Examples
    --------
    >>> hat_function(np.linspace(0, 1, 10))
    array([ 0.        ,  0.95099207,  0.99913557,  0.99999812,  1.        ,
            1.        ,  0.99999812,  0.99913557,  0.95099207,  0.        ])
    """

    a = as_float_array(a)

    return 1 - (2 * a - 1) ** 12


def weighting_function_Debevec1997(
    a: ArrayLike, domain_l: Floating = 0.01, domain_h: Floating = 0.99
) -> NDArray:
    """
    Return given array weighted by *Debevec (1997)* function.

    Parameters
    ----------
    a
        Array to apply the weighting function onto.
    domain_l
        Domain lowest possible value, values less than ``domain_l`` will be set
        to zero.
    domain_h
        Domain highest possible value, values greater than ``domain_h`` will be
        set to zero.

    Returns
    -------
    :class:`numpy.ndarray`
        Weighted array.

    References
    ----------
    :cite:`Debevec1997a`

    Examples
    --------
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
