"""
Weighting Functions
===================

Define the weighting function objects used when generating HDRIs:

-   :func:`colour_hdri.normal_distribution_function`
-   :func:`colour_hdri.hat_function`
-   :func:`colour_hdri.weighting_function_Debevec1997`

References
----------
-   :cite:`Debevec1997a` : Debevec, P. E., & Malik, J. (1997). Recovering high
    dynamic range radiance maps from photographs. Proceedings of the 24th
    Annual Conference on Computer Graphics and Interactive Techniques -
    SIGGRAPH "97, August, 369-378. doi:10.1145/258734.258884
-   :cite:`Mansencal2024` : Mansencal, T. (2024). Double Sigmoid (Anchored).
    Retrieved July 1, 2024, from https://www.desmos.com/calculator/nowptzrt4a
"""

from __future__ import annotations

import numpy as np
from colour.algebra import sdiv, sdiv_mode
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "normal_distribution_function",
    "hat_function",
    "weighting_function_Debevec1997",
    "double_sigmoid_anchored_function",
]


def normal_distribution_function(
    a: ArrayLike, mu: float = 0.5, sigma: float = 0.15
) -> NDArrayFloat:
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


def hat_function(a: ArrayLike) -> NDArrayFloat:
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

    return 1 - (2 * a - 1) ** 12  # pyright: ignore


def weighting_function_Debevec1997(
    a: ArrayLike, domain_l: float = 0.01, domain_h: float = 0.99
) -> NDArrayFloat:
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


def double_sigmoid_anchored_function(
    a: ArrayLike,
    domain_l_in: float = 0.025,
    domain_l_out: float = 0.2,
    domain_h_in: float = 0.8,
    domain_h_out: float = 0.975,
    k: float = 2,
) -> NDArrayFloat:
    """
    Return given array weighted by a double-sigmoid function.

    Parameters
    ----------
    a
        Array to apply the weighting function onto.
    domain_l_in
        Domain lowest possible value, values less than ``domain_l_in`` will be
        set to zero.
    domain_l_out
        Values between ``domain_l_in`` and ``domain_l_out`` will be
        gracefully set to zero by a sigmoid function.
    domain_h_in
        Values between ``domain_h_in`` and ``domain_h_out`` will be
        gracefully set to zero by a sigmoid function.
    domain_h_out
        Domain highest possible value, values greater than ``domain_h_out`` will
        be set to zero.
    k
        Sigmoid function exponentiation factor.

    Returns
    -------
    :class:`numpy.ndarray`
        Weighted array.

    References
    ----------
    :cite:`Mansencal2024`

    Examples
    --------
    >>> double_sigmoid_anchored_function(np.linspace(0, 1, 10))
    array([ 0.        ,  0.48413098,  1.        ,  1.        ,  1.        ,
            1.        ,  1.        ,  1.        ,  0.48413098,  0.        ])
    """

    a = as_float_array(a)

    def anchored_sigmoid_function(
        x: NDArrayFloat, c: float, d: float, k: float
    ) -> NDArrayFloat:
        with sdiv_mode():
            return 1 / (1 + np.power(1 / sdiv(x - c, d - c) - 1, k))

    return np.select(
        [
            a <= domain_l_in,
            np.logical_and(a > domain_l_in, a <= domain_l_out),
            np.logical_and(a > domain_l_out, a < domain_h_in),
            np.logical_and(a >= domain_h_in, a < domain_h_out),
            a >= domain_h_out,
        ],
        [
            0,
            anchored_sigmoid_function(a, domain_l_in, domain_l_out, k),
            1,
            1 - anchored_sigmoid_function(a, domain_h_in, domain_h_out, k),
            0,
        ],
    )
