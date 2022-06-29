"""
Lens Vignette Characterisation & Correction
===========================================

Defines various objects to correct camera lens vignette:

-   :func:`colour_hdri.DataVignetteCharacterisation`
-   :func:`colour_hdri.characterise_vignette_parabolic`
-   :func:`colour_hdri.correct_vignette_parabolic`

References
----------
-   :cite:`Kordecki2016` : Kordecki, A., Palus, H., & Bal, A. (2016). Practical
    vignetting correction method for digital camera with measurement of surface
    luminance distribution. Signal, Image and Video Processing, 10(8),
    1417–1424. doi:10.1007/s11760-016-0941-2
-   :cite:`WonpilYu2004` : Wonpil Yu. (2004). Practical anti-vignetting methods
    for digital cameras. IEEE Transactions on Consumer Electronics, 50(4),
    975–983. doi:10.1109/TCE.2004.1362487
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from colour.algebra import linear_conversion
from colour.hints import (
    ArrayLike,
    Callable,
    Floating,
    Literal,
    NDArray,
    Tuple,
    Union,
    cast,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    MixinDataclassIterable,
    as_float_array,
    tsplit,
    validate_method,
    zeros,
)
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "apply_radial_gradient",
    "parabolic_2D_function",
    "hyperbolic_cosine_2D_function",
    "FunctionVignetteCharacterisation",
    "VIGNETTE_CHARACTERISATION_METHODS",
    "DataVignetteCharacterisation",
    "vignette_center",
    "characterise_vignette",
    "correct_vignette",
]


def apply_radial_gradient(
    image: ArrayLike,
    scale: ArrayLike = (1, 1),
    offset: ArrayLike = (0.5, 0.5),
    intensity: Floating = 1,
    bias: Floating = 1,
    noise: Floating = 0,
) -> NDArray:
    """
    Apply a radial gradient on given image.

    Parameters
    ----------
    image
        Image to apply the radial gradient onto.
    scale
        Radial gradient scale as a ratio of the image height.
    offset
        Radial gradiant offset from a center and as a ratio of image dimensions.
    intensity
        Radial gradient intensity where a value of 1 produces black at the
        top and bottom corners.
    bias
        Power function applied on the gradient.
    noise
        Noise factor.

    Returns
    -------
    :class:`numpy.ndarray`
        Image with radial gradient applied.

    Examples
    --------
    >>> np.around(apply_radial_gradient(np.ones([5, 7])), 3)
    array([[ 0.   ,  0.023,  0.212,  0.286,  0.212,  0.023,  0.   ],
           [ 0.   ,  0.244,  0.511,  0.643,  0.511,  0.244,  0.   ],
           [ 0.   ,  0.333,  0.667,  1.   ,  0.667,  0.333,  0.   ],
           [ 0.   ,  0.244,  0.511,  0.643,  0.511,  0.244,  0.   ],
           [ 0.   ,  0.023,  0.212,  0.286,  0.212,  0.023,  0.   ]])
    """

    image = np.atleast_3d(image)
    scale_x, scale_y = tsplit(scale)
    offset_x, offset_y = tsplit(offset)

    height, width = cast(Tuple, image.shape)[0:2]
    ratio = height / width

    samples_x = np.linspace(-1, 1, height)
    samples_x *= (1 / scale_x) * ratio
    samples_x += offset_x - 0.5
    samples_y = np.linspace(-1, 1, width)
    samples_y *= 1 / scale_y
    samples_y += offset_y - 0.5

    distance = np.sqrt(
        (samples_x**2)[..., np.newaxis] + (samples_y**2)[np.newaxis, ...]
    )

    image *= 1 - distance[..., np.newaxis] * intensity
    image **= bias

    image += np.random.random(image.shape) * noise

    return np.squeeze(np.nan_to_num(np.clip(image, 0, 1)))


def parabolic_2D_function(
    x_y: Tuple,
    a_x2: Floating,
    a_x1: Floating,
    a_x0: Floating,
    a_y2: Floating,
    a_y1: Floating,
    a_y0: Floating,
):
    """
    Evaluate a parabolic 2D function on given coordinate matrices from
    coordinate vectors.

    The parabolic 2D function adopts the following form as given
    by :cite:`Kordecki2016`:

    :math:`I_v(x, y) = \\cfrac{1}{2}(a_{x2}x^2 + a_{x1}x + a_{x0}) + \
\\cfrac{1}{2}(a_{y2}y^2 + a_{y1}y + a_{y0})`

    Parameters
    ----------
    x_y
        Coordinate matrices from coordinate vectors to evaluate the parabolic
        2d function on. The coordinate matrices can be generated with the
        :func:`numpy.meshgrid` definition.
    a_x2
        Coefficient :math:`a_{x2}` for the parabolic equation.
    a_x1
        Coefficient :math:`a_{x1}` for the parabolic equation.
    a_x0
        Coefficient :math:`a_{x0}` for the parabolic equation.
    a_y2
        Coefficient :math:`a_{y2}` for the parabolic equation.
    a_y1
        Coefficient :math:`a_{y1}` for the parabolic equation.
    a_y0
        Coefficient :math:`a_{y0}` for the parabolic equation.

    Returns
    -------
    :class:`numpy.ndarray`
        Coordinate matrices with evaluated parabolic 2D function.

    References
    ----------
    :cite:`Kordecki2016`

    Examples
    --------
    >>> x_1, y_1 = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    >>> parabolic_2D_function(  # doctest: +ELLIPSIS
    ...     (x_1, y_1), -0.5, 0, 1, -0.5, 0, 1)
    array([[ 1.        ,  0.9722222...,  0.8888888...,  0.75      ],
           [ 0.9375    ,  0.9097222...,  0.8263888...,  0.6875    ],
           [ 0.75      ,  0.7222222...,  0.6388888...,  0.5       ]])
    """

    x, y = x_y

    I_v = (a_x2 * x**2 + a_x1 * x + a_x0) / 2
    I_v += (a_y2 * y**2 + a_y1 * y + a_y0) / 2

    return I_v


def hyperbolic_cosine_2D_function(
    x_y: Tuple,
    r_x: Floating,
    x_0: Floating,
    r_y: Floating,
    y_0: Floating,
    c: Floating,
):
    """
    Evaluate a hyperbolic cosine 2D function on given coordinate matrices from
    coordinate vectors.

    The hyperbolic cosine 2D function adopts the following form:

    :math:`I_v(x, y) = 1 - (cosh(r_x * (x - x_0)) * cosh(r_y * (y - y_0))) + c`

    Parameters
    ----------
    x_y
        Coordinate matrices from coordinate vectors to evaluate the parabolic
        2d function on. The coordinate matrices can be generated with the
        :func:`numpy.meshgrid` definition.
    r_x
        Coefficient :math:`r_x` for the hyperbolic cosine equation.
    x_0
        Coefficient :math:`x_0` for the hyperbolic cosine equation.
    r_y
        Coefficient :math:`r_y` for the hyperbolic cosine equation.
    y_0
        Coefficient :math:`y_0` for the hyperbolic cosine equation.
    c_y
        Coefficient :math:`c_y` for the hyperbolic cosine equation.
    c
        Coefficient :math:`c` for the hyperbolic cosine equation.

    Returns
    -------
    :class:`numpy.ndarray`
        Coordinate matrices with evaluated hyperbolic cosine 2D function.

    References
    ----------
    :cite:`WonpilYu2004`

    Examples
    --------
    >>> x_1, y_1 = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    >>> hyperbolic_cosine_2D_function(  # doctest: +ELLIPSIS
    ...     (x_1, y_1), 1, -0.5, 1, -0.5, 1)
    array([[ 1.       ...,  0.9439281...,  0.7694244...,  0.4569193...],
           [ 0.8723740...,  0.8091459...,  0.6123710...,  0.2599822...],
           [ 0.4569193...,  0.3703959...,  0.1011226..., -0.3810978...]])
    """

    x, y = x_y

    x = linear_conversion(x, (0, 1), (-0.5, 0.5))
    y = linear_conversion(y, (0, 1), (-0.5, 0.5))

    I_v = 1 - (np.cosh(r_x * (x - x_0)) * np.cosh(r_y * (y - y_0))) + c

    return I_v


@dataclass
class FunctionVignetteCharacterisation(MixinDataclassIterable):
    """
    Define a vignette characterisation function and the required data for
    fitting it to an image.

    Parameters
    ----------
    function
        Vignette characterisation function.
    p0
        Initial guess for the function fitting, passed to
        :func:`scipy.optimize.curve_fit` definition.
    bounds
        Lower and upper bounds for the function fitting, passed to
        :func:`scipy.optimize.curve_fit` definition.
    """

    function: Callable
    p0: NDArray
    bounds: NDArray


VIGNETTE_CHARACTERISATION_METHODS: CaseInsensitiveMapping = (
    CaseInsensitiveMapping(
        {
            "Parabolic": FunctionVignetteCharacterisation(
                parabolic_2D_function,
                np.array([0, 0, 1, 0, 0, 1]),
                np.array(
                    [
                        (-5.0, -0.5, 0.9, -5.0, -0.5, 0.9),
                        (+0.0, +0.5, 1.1, +0.0, +0.5, 1.1),
                    ]
                ),
            ),
            "Hyperbolic Cosine": FunctionVignetteCharacterisation(
                hyperbolic_cosine_2D_function,
                np.array([1, 0, 1, 0, 0]),
                np.array(
                    [
                        (0.5, -1.0, 0.5, -1.0, 0.0),
                        (5.0, +0.0, 5.0, +0.0, 1.5),
                    ]
                ),
            ),
        }
    )
)


@dataclass
class DataVignetteCharacterisation(MixinDataclassIterable):
    """
    Define the data of a vignette characterisation process.

    Parameters
    ----------
    coefficients
        Vignette characterisation function coefficients.
    center
        Image center
    """

    coefficients: ArrayLike
    center: ArrayLike


def vignette_center(image: ArrayLike, threshold: Floating = 0.99) -> NDArray:
    """
    Return the vignette center for given image.

    Parameters
    ----------
    image
        Vignette image to return the center of.
    threshold
        Pixels threshold before finding the vignette center.

    Examples
    --------
    >>> vignette_center(  # doctest: +ELLIPSIS
    ...     apply_radial_gradient(np.ones([5, 7, 3])))
    array([ 0.4       ,  0.4285714...])
    """

    image = as_float_array(image)
    shape_x, shape_y, _ = image.shape

    L = np.median(image, axis=-1)

    center = zeros(L.shape)
    center[L > np.max(L) * threshold] = 1

    return center_of_mass(center) / as_float_array([shape_x, shape_y])


def characterise_vignette(
    image: ArrayLike,
    method: Union[
        Literal["Parabolic", "Hyperbolic Cosine"], str
    ] = "Parabolic",
) -> DataVignetteCharacterisation:
    """
    Characterise the vignette from given image using a given method.

    Parameters
    ----------
    image
        Image to characterise the vignette of.
    method
        Characterisation method.

    Returns
    -------
    :class:`tuple`
        Function coefficients and image center.

    Examples
    --------
    >>> characterise_vignette(  # doctest: +ELLIPSIS
    ...     apply_radial_gradient(np.ones([5, 7])))
    DataVignetteCharacterisation(coefficients=array([[-5.        ,  \
0.5       ,  0.9       , -4.4699758...,  0.5       ,
             0.9       ]]), center=array([ 0.4       ,  0.4285714...]))
    """

    image = np.atleast_3d(image)
    method = validate_method(method, VIGNETTE_CHARACTERISATION_METHODS)

    function, p0, bounds = VIGNETTE_CHARACTERISATION_METHODS[method].values

    height, width, channels = image.shape

    x_1, y_1 = np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(0, 1, height),
    )

    center = vignette_center(image)

    coefficients = []
    for i in range(channels):
        coefficients.append(
            curve_fit(
                function,
                (np.ravel(x_1 - center[0]), np.ravel(y_1 - center[1])),
                np.ravel(np.nan_to_num(image[..., i])),
                p0=p0,
                bounds=bounds,
            )[0]
        )

    return DataVignetteCharacterisation(as_float_array(coefficients), center)


def correct_vignette(
    image: ArrayLike,
    characterisation_data: DataVignetteCharacterisation,
    method: Union[
        Literal["Parabolic", "Hyperbolic Cosine"], str
    ] = "Parabolic",
) -> NDArray:
    """
    Correct the vignette on given image using given coefficients for the
    parabolic function.

    Parameters
    ----------
    image
        Image to correct the vignette of.
    characterisation_data
        Vignette characterisation data for given method.
    method
        Correction method.

    Returns
    -------
    :class:`numpy.ndarray`
        Vignette corrected image.

    Examples
    --------
    >>> image = apply_radial_gradient(np.ones([5, 7]))
    >>> characterisation_data = characterise_vignette(image)
    >>> np.around(correct_vignette(image, characterisation_data), 3)
    ... # doctest: +ELLIPSIS
    array([[-0.   ,  0.122,  0.597,  0.747,  0.781,  1.08 , -0.   ],
           [ 0.   ,  0.413,  0.676,  0.82 ,  0.76 ,  0.576,  0.   ],
           [ 0.   ,  0.468,  0.759,  1.103,  0.838,  0.611,  0.   ],
           [ 0.   ,  0.439,  0.709,  0.858,  0.801,  0.628, -0.   ],
           [-0.   ,  0.193,  0.742,  0.913,  1.049, -0.477, -0.   ]])
    """

    image = np.copy(np.atleast_3d(image))

    coefficients, center = characterisation_data.values

    method = validate_method(method, VIGNETTE_CHARACTERISATION_METHODS)

    vignette_characterisation_function = VIGNETTE_CHARACTERISATION_METHODS[
        method
    ]

    height, width, channels = image.shape

    x_1, y_1 = np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(0, 1, height),
    )

    for i in range(channels):
        image[..., i] /= vignette_characterisation_function.function(
            (x_1 - center[0], y_1 - center[1]), *coefficients[i]
        )

    return np.squeeze(image)
