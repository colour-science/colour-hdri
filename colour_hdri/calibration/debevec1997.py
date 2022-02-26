"""
Debevec (1997) Camera Response Function Computation
===================================================

Defines the *Debevec (1997)* camera responses computation objects:

-   :func:`colour_hdri.g_solve`
-   :func:`colour_hdri.camera_response_functions_Debevec1997`

See Also
--------
`Colour - HDRI - Examples: Merge from Low Dynamic Range Files Jupyter Notebook
<https://github.com/colour-science/colour-hdri/\
blob/master/colour_hdri/examples/examples_merge_from_ldr_files.ipynb>`__

References
----------
-   :cite:`Debevec1997a` : Debevec, P. E., & Malik, J. (1997). Recovering high
    dynamic range radiance maps from photographs. Proceedings of the 24th
    Annual Conference on Computer Graphics and Interactive Techniques -
    SIGGRAPH "97, August, 369-378. doi:10.1145/258734.258884
"""

from __future__ import annotations

import numpy as np
from functools import partial

from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Callable,
    Dict,
    Floating,
    Integer,
    NDArray,
    Optional,
    Tuple,
)
from colour.utilities import as_float_array, as_int_array, tstack

from colour_hdri.exposure import average_luminance
from colour_hdri.generation import weighting_function_Debevec1997
from colour_hdri.sampling import samples_Grossberg2003
from colour_hdri.utilities import ImageStack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "g_solve",
    "extrapolating_function_polynomial",
    "camera_response_functions_Debevec1997",
]


def g_solve(
    Z: ArrayLike,
    B: ArrayLike,
    l_s: Floating = 30,
    w: Callable = weighting_function_Debevec1997,
    n: Integer = 256,
) -> Tuple[NDArray, NDArray]:
    """
    Given a set of pixel values observed for several pixels in several images
    with different exposure times, this function returns the imaging system's
    response function :math:`g` as well as the log film irradiance values
    :math:`lE` for the observed pixels.

    Parameters
    ----------
    Z
        Set of pixel values observed for several pixels in several images.
    B
        Log :math:`\\Delta t`, or log shutter speed for images.
    l_s
        :math:`\\lambda` smoothing term.
    w
        Weighting function :math:`w`.
    n
        :math:`n` constant.

    Returns
    -------
    :class:`tuple`
        Camera response functions :math:`g(z)` and log film irradiance values
        :math:`lE`.

    References
    ----------
    :cite:`Debevec1997a`
    """

    Z = as_int_array(Z)
    B = as_float_array(B)

    Z_x, Z_y = Z.shape

    A = np.zeros((Z_x * Z_y + n + 1, n + Z_x))
    b = np.zeros((A.shape[0], 1))
    w_n = w(np.linspace(0, 1, n))

    k = 0
    for i in np.arange(Z_x):
        for j in np.arange(Z_y):
            Z_c = Z[i, j]
            w_ij = w_n[Z_c]
            A[k, Z_c] = w_ij
            A[k, n + i] = -w_ij
            b[k] = w_ij * B[j]
            k += 1

    A[k, np.int_(n / 2)] = 1
    k += 1

    for i in np.arange(1, n - 1, 1):
        A[k, i - 1] = l_s * w_n[i]
        A[k, i + 0] = l_s * w_n[i] * -2
        A[k, i + 1] = l_s * w_n[i]
        k += 1

    x = np.squeeze(np.linalg.lstsq(A, b, rcond=None)[0])

    g = x[0:n]
    lE = x[n : x.shape[0]]

    return g, lE


def extrapolating_function_polynomial(
    crfs: ArrayLike,
    weighting_function: Callable,
    degree: Integer = 7,
    **kwargs: Any,
) -> NDArray:
    """
    Polynomial extrapolating function used to handle zero-weighted data of
    given camera response functions.

    The extrapolation occurs where the weighting function masks fully the
    camera response functions, e.g. at both ends for *Debevec (1997)*.

    Parameters
    ----------
    crfs
        Camera response functions :math:`g(z)`.
    weighting_function
        Weighting function :math:`w`.
    degree
        Degree of the extrapolating function polynomial.

    Other Parameters
    ----------------
    kwargs
        Keyword arguments.

    Returns
    -------
    :class:`numpy.ndarray`
        Extrapolated camera response functions :math:`g(z)`.
    """

    crfs = as_float_array(crfs)

    samples = np.linspace(0, 1, crfs.shape[0])
    mask = ~(weighting_function(samples) == 0)

    for x in range(crfs.shape[-1]):
        coefficients = np.polyfit(samples[mask], crfs[mask, x], degree)
        polynomial = np.poly1d(coefficients)
        crfs[~mask, x] = polynomial(samples[~mask])

    return crfs


def camera_response_functions_Debevec1997(
    image_stack: ImageStack,
    sampling_function: Callable = samples_Grossberg2003,
    sampling_function_kwargs: Optional[Dict] = None,
    weighting_function: Callable = weighting_function_Debevec1997,
    weighting_function_kwargs: Optional[Dict] = None,
    extrapolating_function: Callable = extrapolating_function_polynomial,
    extrapolating_function_kwargs: Optional[Dict] = None,
    l_s: Floating = 30,
    n: Integer = 256,
    normalise: Boolean = True,
) -> NDArray:
    """
    Return the camera response functions for given image stack using
    *Debevec (1997)* method.

    Image channels are sampled with :math:`s` sampling function and the output
    samples are passed to :func:`colour_hdri.g_solve`.

    Parameters
    ----------
    image_stack
        Stack of single channel or multi-channel floating point images.
    sampling_function
        Sampling function :math:`s`.
    sampling_function_kwargs
        Arguments to use when calling the sampling function.
    weighting_function
        Weighting function :math:`w`.
    weighting_function_kwargs
        Arguments to use when calling the weighting function.
    extrapolating_function
        Extrapolating function used to handle zero-weighted data.
    extrapolating_function_kwargs
        Arguments to use when calling the extrapolating function.
    l_s
        :math:`\\lambda` smoothing term.
    n
        :math:`n` constant.
    normalise
        Enables the camera response functions normalisation.

    Returns
    -------
    :class:`numpy.ndarray`
        Camera response functions :math:`g(z)`.

    References
    ----------
    :cite:`Debevec1997a`
    """

    if sampling_function_kwargs is None:
        sampling_function_kwargs = {}

    if weighting_function_kwargs is None:
        weighting_function_kwargs = {}

    if extrapolating_function_kwargs is None:
        extrapolating_function_kwargs = {}

    s_o = sampling_function(image_stack.data, **sampling_function_kwargs)

    L_l = np.log(
        1
        / average_luminance(
            image_stack.f_number, image_stack.exposure_time, image_stack.iso
        )
    )

    w = partial(weighting_function, **weighting_function_kwargs)

    g_c = [
        g_solve(s_o[..., x], L_l, l_s, w, n)[0] for x in range(s_o.shape[-1])
    ]
    crfs = np.exp(tstack(np.array(g_c)))

    if extrapolating_function is not None:
        crfs = extrapolating_function(crfs, w, **extrapolating_function_kwargs)

    if normalise:
        crfs /= np.max(crfs, axis=0)

    return crfs
