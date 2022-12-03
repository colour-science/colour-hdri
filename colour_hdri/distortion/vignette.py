"""
Lens Vignette Characterisation & Correction
===========================================

Defines various objects to correct camera lens vignette:

-   :func:`colour_hdri.distortion.apply_radial_gradient`
-   :func:`colour_hdri.distortion.parabolic_2D_function`
-   :func:`colour_hdri.distortion.hyperbolic_cosine_2D_function`
-   :func:`colour_hdri.distortion.DataVignetteCharacterisation`
-   :func:`colour_hdri.distortion.characterise_vignette_2D_function`
-   :func:`colour_hdri.distortion.correct_vignette_2D_function`
-   :func:`colour_hdri.distortion.characterise_vignette_bivariate_spline`
-   :func:`colour_hdri.distortion.correct_vignette_bivariate_spline`
-   :func:`colour_hdri.distortion.characterise_vignette_RBF`
-   :func:`colour_hdri.distortion.correct_vignette_RBF`
-   :func:`colour_hdri.VIGNETTE_CHARACTERISATION_METHODS`
-   :func:`colour_hdri.characterise_vignette`
-   :func:`colour_hdri.VIGNETTE_CORRECTION_METHODS`
-   :func:`colour_hdri.correct_vignette`

References
----------
-   :cite:`Kordecki2016` : Kordecki, A., Palus, H., & Bal, A. (2016). Practical
    vignetting correction function for digital camera with measurement of surface
    luminance distribution. Signal, Image and Video Processing, 10(8),
    1417-1424. doi:10.1007/s11760-016-0941-2
-   :cite:`WonpilYu2004` : Wonpil Yu. (2004). Practical anti-vignetting methods
    for digital cameras. IEEE Transactions on Consumer Electronics, 50(4),
    975-983. doi:10.1109/TCE.2004.1362487
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RBFInterpolator, RectBivariateSpline
from scipy.ndimage import center_of_mass, gaussian_filter
from scipy.optimize import curve_fit

from colour.algebra import (
    LinearInterpolator,
    linear_conversion,
    polar_to_cartesian,
)
from colour.hints import (
    ArrayLike,
    Callable,
    Floating,
    Integer,
    Literal,
    NDArray,
    Tuple,
    Union,
    cast,
)
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassIterable,
    as_float_array,
    as_int_array,
    ones,
    tsplit,
    tstack,
    validate_method,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "apply_radial_gradient",
    "vignette_principal_point",
    "parabolic_2D_function",
    "hyperbolic_cosine_2D_function",
    "FunctionVignetteCharacterisation",
    "VIGNETTE_CHARACTERISATION_2D_FUNCTIONS",
    "DataVignetteCharacterisation",
    "characterise_vignette_2D_function",
    "correct_vignette_2D_function",
    "characterise_vignette_bivariate_spline",
    "correct_vignette_bivariate_spline",
    "radial_sampling_function",
    "vignette_sampling_coordinates",
    "characterise_vignette_RBF",
    "correct_vignette_RBF",
    "VIGNETTE_CHARACTERISATION_METHODS",
    "characterise_vignette",
    "VIGNETTE_CORRECTION_METHODS",
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
        Radial gradiant offset from the image center and as a ratio of image
        dimensions.
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
        (samples_x**2)[..., None] + (samples_y**2)[None, ...]
    )

    image *= 1 - distance[..., None] * intensity
    image **= bias

    image += np.random.random(image.shape) * noise

    return np.squeeze(np.nan_to_num(np.clip(image, 0, 1)))


def vignette_principal_point(
    image: ArrayLike, threshold: Floating = 0.99
) -> NDArray:
    """
    Return the vignette principal point for given image.

    Parameters
    ----------
    image
        Vignette image to return the principal point of.
    threshold
        Pixels threshold before finding the vignette principal point.

    Returns
    -------
    :class:`numpy.ndarray`
        Vignette principal point.

    Examples
    --------
    >>> vignette_principal_point(  # doctest: +ELLIPSIS
    ...     apply_radial_gradient(np.ones([5, 7, 3]))
    ... )
    array([ 0.4       ,  0.4285714...])
    """

    image = as_float_array(image)
    shape_x, shape_y, _ = image.shape

    M = np.median(image, axis=-1)

    thresholded = zeros(M.shape)
    thresholded[M > np.max(M) * threshold] = 1

    return center_of_mass(thresholded) / as_float_array([shape_x, shape_y])


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
    ...     (x_1, y_1), -0.5, 0, 1, -0.5, 0, 1
    ... )
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
    ...     (x_1, y_1), 1, -0.5, 1, -0.5, 1
    ... )
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


VIGNETTE_CHARACTERISATION_2D_FUNCTIONS: CanonicalMapping = CanonicalMapping(
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
VIGNETTE_CHARACTERISATION_2D_FUNCTIONS.__doc__ = """
Supported vignette characterisation 2D functions.

References
----------
:cite:`Kordecki2016`, :cite:`WonpilYu2004`
"""


@dataclass
class DataVignetteCharacterisation(
    MixinDataclassIterable
):  # noqa: D405,D407,D410,D411,D414
    """
    Define the data of a vignette characterisation process.

    Parameters
    ----------
    parameters
        Vignette characterisation parameters.
    principal_point
        Vignette principal point.
    """

    parameters: ArrayLike
    principal_point: ArrayLike


def characterise_vignette_2D_function(
    image: ArrayLike,
    function: Union[
        Literal["Parabolic", "Hyperbolic Cosine"], str
    ] = "Parabolic",
) -> DataVignetteCharacterisation:
    """
    Characterise the vignette of given image using a given 2D function.

    Parameters
    ----------
    image
        Image to characterise the vignette of.
    function
        Characterisation function.

    Returns
    -------
    :class:`DataVignetteCharacterisation`
        Vignette characterisation.

    Examples
    --------
    >>> characterise_vignette_2D_function(  # doctest: +ELLIPSIS
    ...     apply_radial_gradient(np.ones([5, 7]))
    ... )
    DataVignetteCharacterisation(parameters=array([[-5.        ,  0.5       ,  \
0.9       , -4.4699758...,  0.5       ,
             0.9       ]]), principal_point=array([ 0.4       ,  0.4285714...]))
    """

    image = np.atleast_3d(image)
    function = validate_method(
        function,
        VIGNETTE_CHARACTERISATION_2D_FUNCTIONS,
        '"{0}" function is invalid, it must be one of {1}!',
    )

    (
        vignette_characterisation_function,
        p0,
        bounds,
    ) = VIGNETTE_CHARACTERISATION_2D_FUNCTIONS[function].values

    height, width, channels = image.shape

    x_1, y_1 = np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(0, 1, height),
    )

    principal_point = vignette_principal_point(image)

    parameters = []
    for i in range(channels):
        parameters.append(
            curve_fit(
                vignette_characterisation_function,
                (
                    np.ravel(x_1 - principal_point[0]),
                    np.ravel(y_1 - principal_point[1]),
                ),
                np.ravel(np.nan_to_num(image[..., i])),
                p0=p0,
                bounds=bounds,
            )[0]
        )

    return DataVignetteCharacterisation(
        as_float_array(parameters), principal_point
    )


def correct_vignette_2D_function(
    image: ArrayLike,
    characterisation_data: DataVignetteCharacterisation,
    function: Union[
        Literal["Parabolic", "Hyperbolic Cosine"], str
    ] = "Parabolic",
) -> NDArray:
    """
    Correct the vignette of given image using given characterisation for a
    2D function.

    Parameters
    ----------
    image
        Image to correct the vignette of.
    characterisation_data
        Vignette characterisation data for given function.
    function
        Correction function.

    Returns
    -------
    :class:`numpy.ndarray`
        Vignette corrected image.

    Examples
    --------
    >>> image = apply_radial_gradient(np.ones([5, 7]))
    >>> characterisation_data = characterise_vignette_2D_function(image)
    >>> np.around(
    ...     correct_vignette_2D_function(image, characterisation_data), 3
    ... )
    array([[-0.   ,  0.122,  0.597,  0.747,  0.781,  1.08 , -0.   ],
           [ 0.   ,  0.413,  0.676,  0.82 ,  0.76 ,  0.576,  0.   ],
           [ 0.   ,  0.468,  0.759,  1.103,  0.838,  0.611,  0.   ],
           [ 0.   ,  0.439,  0.709,  0.858,  0.801,  0.628, -0.   ],
           [-0.   ,  0.193,  0.742,  0.913,  1.049, -0.477, -0.   ]])
    """

    image = np.copy(np.atleast_3d(image))

    function = validate_method(
        function,
        VIGNETTE_CHARACTERISATION_2D_FUNCTIONS,
        '"{0}" function is invalid, it must be one of {1}!',
    )

    vignette_characterisation_function = (
        VIGNETTE_CHARACTERISATION_2D_FUNCTIONS[function]
    )

    parameters, principal_point = characterisation_data.values

    height, width, channels = image.shape

    x_1, y_1 = np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(0, 1, height),
    )

    for i in range(channels):
        image[..., i] /= vignette_characterisation_function.function(
            (x_1 - principal_point[0], y_1 - principal_point[1]),
            *parameters[i],
        )

    return np.squeeze(image)


def characterise_vignette_bivariate_spline(
    image: ArrayLike,
    pre_denoise_sigma: Floating = 6,
    post_denoise_sigma: Floating = 1,
    samples: Integer = 50,
    degree: Integer = 3,
) -> DataVignetteCharacterisation:
    """
    Characterise the vignette of given image using a bivariate spline.

    Parameters
    ----------
    image
        Image to characterise the vignette of.
    pre_denoise_sigma
        Standard deviation of the gaussian filtering kernel applied on the
        image.
    post_denoise_sigma
        Standard deviation of the gaussian filtering kernel applied on the
        resampled image at given ``samples`` count.
    samples
        Samples count of the resampled image on the long edge.
    degree
        Degree of the bivariate spline.

    Returns
    -------
    :class:`DataVignetteCharacterisation`
        Vignette characterisation.

    Examples
    --------
    >>> parameters, principal_point = characterise_vignette_bivariate_spline(
    ...     apply_radial_gradient(np.ones([300, 400]))
    ... ).values
    >>> parameters.shape
    (37, 50, 1)
    >>> principal_point  # doctest: +ELLIPSIS
    array([ 0.4983333...,  0.49875   ])
    """

    image = np.copy(np.atleast_3d(image))

    principal_point = vignette_principal_point(image)

    height, width, channels = image.shape
    ratio = samples / max(height, width)
    height_n, width_n = int(height * ratio), int(width * ratio)

    x_1, y_1 = np.linspace(0, 1, height), np.linspace(0, 1, width)
    x_1_n, y_1_n = np.linspace(0, 1, height_n), np.linspace(0, 1, width_n)

    # NOTE: Here "parameters" represent a lower resolution version of the
    # image, i.e. the "I_v" function directly.
    parameters = zeros((height_n, width_n, channels))

    for i in range(channels):
        image[..., i] = gaussian_filter(
            image[..., i],
            pre_denoise_sigma,
            truncate=pre_denoise_sigma,
            mode="nearest",
        )

        interpolator = RectBivariateSpline(
            x_1, y_1, image[..., i], kx=degree, ky=degree
        )

        parameters[..., i] = interpolator(x_1_n, y_1_n)

        parameters[..., i] = gaussian_filter(
            parameters[..., i],
            post_denoise_sigma,
            truncate=pre_denoise_sigma,
            mode="nearest",
        )

    return DataVignetteCharacterisation(parameters, principal_point)


def correct_vignette_bivariate_spline(
    image: ArrayLike,
    characterisation_data: DataVignetteCharacterisation,
    degree: Integer = 3,
) -> NDArray:
    """
    Correct the vignette of given image using given characterisation for a
    bivariate spline.

    Parameters
    ----------
    image
        Image to correct the vignette of.
    characterisation_data
        Vignette characterisation data for given function.
    degree
        Degree of the bivariate spline.

    Returns
    -------
    :class:`numpy.ndarray`
        Vignette corrected image.

    Examples
    --------
    >>> image = apply_radial_gradient(np.ones([5, 7]))
    >>> characterisation_data = characterise_vignette_bivariate_spline(image)
    >>> np.around(
    ...     correct_vignette_bivariate_spline(image, characterisation_data), 3
    ... )
    array([[  0.   ,   0.345,   3.059,   4.072,   3.059,   0.345,   0.   ],
           [  0.   ,   3.624,   7.304,   9.058,   7.304,   3.624,   0.   ],
           [  0.   ,   4.936,   9.481,  14.032,   9.481,   4.936,   0.   ],
           [  0.   ,   3.624,   7.304,   9.058,   7.304,   3.624,   0.   ],
           [  0.   ,   0.345,   3.059,   4.072,   3.059,   0.345,   0.   ]])
    """

    image = np.copy(np.atleast_3d(image))

    parameters, principal_point = characterisation_data.values

    height, width, channels = image.shape
    height_I_v, width_I_v, channels_I_v = parameters.shape

    x_1, y_1 = np.linspace(0, 1, height), np.linspace(0, 1, width)
    x_I_v, y_I_v = np.linspace(0, 1, height_I_v), np.linspace(0, 1, width_I_v)

    for i in range(channels):
        interpolator = RectBivariateSpline(
            x_I_v, y_I_v, parameters[..., i], kx=degree, ky=degree
        )

        image[..., i] /= interpolator(x_1, y_1)

    return np.squeeze(image)


def radial_sampling_function(
    samples_rho: Integer = 7,
    samples_phi: Integer = 21,
    radius: Floating = 1,
    radial_bias: Floating = 1,
) -> NDArray:
    """
    Return a series of radial samples.

    Parameters
    ----------
    samples_rho
        Sample count along the radial coordinate.
    samples_phi
        Sample count along the angular coordinate.
    radius
        Sample distribution radius.
    radial_bias
        Sample distribution bias, i.e. an exponent affecting the radial
        distribution.

    Returns
    -------
    :class:`numpy.ndarray`
        Radial samples.

    Examples
    --------
    >>> radial_sampling_function().shape
    (21, 7, 2)
    """

    rho, phi = np.meshgrid(
        np.linspace(0, radius, samples_rho) ** radial_bias,
        np.linspace(-np.pi, np.pi, samples_phi),
    )

    return polar_to_cartesian(tstack([rho, phi]))


def vignette_sampling_coordinates(
    principal_point: ArrayLike = np.array([0.5, 0.5]),
    aspect_ratio: Floating = 1,
    diagonal_samples: Integer = 10,
    diagonal_selection: Integer = 2,
    edge_samples: Integer = 10,
    samples_rho: Integer = 7,
    samples_phi: Integer = 21,
    radius: Floating = 0.9,
    radial_bias: Floating = 1,
) -> NDArray:
    """
    Return a series of sampling coordinates appropriate for radial basis
    function (RBF) interpolation of a vignette function.

    Parameters
    ----------
    principal_point
        Principal point of the vignette function to sample.
    aspect_ratio
        Aspect ratio of the image storing the vignette function to sample.
    diagonal_samples
        Sample count along the diagonals.
    diagonal_selection
        Sample count to retain along the diagonals ends. Given a series of 6
        ``diagonal_samples`` as follows: `[0, 1, 2, 3, 4, 5]`, a
        ``diagonal_selection`` of 2 would retain the following samples:
        `[0, 1, 4, 5]`.
    edge_samples
        Sample count along the edges.
    samples_rho
        Sample count along the radial coordinate.
    samples_phi
        Sample count along the angular coordinate.
    radius
        Sample distribution radius.
    radial_bias
        Sample distribution bias, i.e. an exponent affecting the radial
        distribution.

    Returns
    -------
    :class:`numpy.ndarray`
        Radial samples.

    Examples
    --------
    >>> vignette_sampling_coordinates().shape
    (187, 2)
    """

    principal_point = as_float_array(principal_point)

    samples = []

    diagonal = np.linspace(0, 1, diagonal_samples)
    diagonal = np.hstack(
        [diagonal[1:diagonal_selection], diagonal[-diagonal_selection:-1]]
    )
    samples.append(tstack([diagonal, diagonal]))
    samples.append(tstack([diagonal, 1 - diagonal]))

    edge = np.linspace(0, 1, edge_samples)
    samples.append(tstack([edge, zeros(edge_samples)]))
    samples.append(tstack([edge, ones(edge_samples)]))
    samples.append(tstack([zeros(edge_samples), edge])[1:-1])
    samples.append(tstack([ones(edge_samples), edge])[1:-1])

    coordinates = np.vstack(samples)

    coordinates[..., 0] = LinearInterpolator(
        [0, 0.5, 1], [0, principal_point[0], 1]
    )(coordinates[..., 0])
    coordinates[..., 1] = LinearInterpolator(
        [0, 0.5, 1], [0, principal_point[1], 1]
    )(coordinates[..., 1])

    radial_samples = radial_sampling_function(
        samples_rho,
        samples_phi,
        1 + (np.max(principal_point - 0.5) * 2),
        radial_bias,
    )
    # NOTE: Some randomisation is required to avoid a
    # "LinAlgError: Singular matrix" exception raised by
    # "scipy.interpolate.RBFInterpolator" definition.
    radial_samples += (
        np.random.default_rng(8).random(radial_samples.shape) - 0.5
    ) / 1000
    radial_samples = np.reshape(radial_samples / (2 * 1 / radius), [-1, 2])
    radial_samples[..., 1] *= aspect_ratio
    radial_samples += principal_point

    coordinates = np.vstack([coordinates, radial_samples])

    coordinates = coordinates[
        np.logical_and(
            np.all(coordinates >= 0, axis=-1),
            np.all(coordinates <= 1, axis=-1),
        )
    ]

    return coordinates


def characterise_vignette_RBF(
    image: ArrayLike, denoise_sigma: Floating = 6
) -> DataVignetteCharacterisation:
    """
    Characterise the vignette of given image using a series of sampling
    coordinates appropriate for radial basis function (RBF) interpolation of a
    vignette function.

    Parameters
    ----------
    image
        Image to characterise the vignette of.
    denoise_sigma
        Standard deviation of the gaussian filtering kernel applied on the
        image.

    Returns
    -------
    :class:`DataVignetteCharacterisation`
        Vignette characterisation.

    Examples
    --------
    >>> parameters, principal_point = characterise_vignette_RBF(
    ...     apply_radial_gradient(np.ones([300, 400]))
    ... ).values
    >>> parameters.shape
    (180, 1)
    >>> principal_point  # doctest: +ELLIPSIS
    array([ 0.4983333...,  0.49875   ])
    """

    image = np.copy(np.atleast_3d(image))

    height, width, channels = image.shape

    principal_point = vignette_principal_point(image)

    sampling_coordinates = vignette_sampling_coordinates(
        principal_point, width / height
    )
    x_indices = as_int_array(sampling_coordinates[..., 0] * (height - 1))
    y_indices = as_int_array(sampling_coordinates[..., 1] * (width - 1))

    parameters = []
    for i in range(channels):
        filtered = gaussian_filter(
            image[..., i], denoise_sigma, truncate=denoise_sigma
        )
        parameters.append(filtered[x_indices, y_indices])

    return DataVignetteCharacterisation(
        np.transpose(parameters), principal_point
    )


def correct_vignette_RBF(
    image: ArrayLike,
    characterisation_data: DataVignetteCharacterisation,
    smoothing: Floating = 0.001,
    kernel: Literal[
        "linear",
        "thin_plate_spline",
        "cubic",
        "quintic",
        "multiquadric",
        "inverse_multiquadric",
        "inverse_quadratic",
        "gaussian",
    ] = "cubic",
    epsilon: Floating = 1,
) -> NDArray:
    """
    Correct the vignette of given image using given characterisation for
    radial basis function (RBF) interpolation.

    Parameters
    ----------
    image
        Image to correct the vignette of.
    characterisation_data
        Vignette characterisation data for given function.
    smoothing
        Smoothing parameter, see :class:`scipy.interpolate.RBFInterpolator`
        class.
    kernel
         Type of RBF, see :class:`scipy.interpolate.RBFInterpolator` class.
    epsilon
        Shape parameter that scales the input to the RBF, see
        :class:`scipy.interpolate.RBFInterpolator` class.

    Returns
    -------
    :class:`numpy.ndarray`
        Vignette corrected image.

    Examples
    --------
    >>> image = apply_radial_gradient(np.ones([5, 7]))
    >>> characterisation_data = characterise_vignette_RBF(image)
    >>> np.around(correct_vignette_RBF(image, characterisation_data), 3)
    array([[ 0.   ,  0.091,  0.841,  1.134,  0.841,  0.091,  0.   ],
           [ 0.   ,  0.967,  2.03 ,  2.552,  2.03 ,  0.967,  0.   ],
           [ 0.   ,  1.323,  2.647,  3.97 ,  2.647,  1.323,  0.   ],
           [ 0.   ,  0.967,  2.03 ,  2.552,  2.03 ,  0.967,  0.   ],
           [ 0.   ,  0.091,  0.841,  1.134,  0.841,  0.091,  0.   ]])
    """

    image = np.copy(np.atleast_3d(image))

    height, width, channels = image.shape

    parameters, principal_point = characterisation_data.values

    sampling_coordinates = vignette_sampling_coordinates(
        principal_point, width / height
    )
    x_1, y_1 = np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(0, 1, height),
    )

    for i in range(channels):
        interpolator = RBFInterpolator(
            sampling_coordinates,
            parameters[..., i],
            kernel=kernel,
            smoothing=smoothing,
            epsilon=epsilon,
        )

        I_v = interpolator(tstack([y_1, x_1]).reshape([-1, 2])).reshape(
            height, width
        )

        image[..., i] /= I_v

    return np.squeeze(image)


VIGNETTE_CHARACTERISATION_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "2D Function": characterise_vignette_2D_function,
        "Bivariate Spline": characterise_vignette_bivariate_spline,
        "RBF": characterise_vignette_RBF,
    }
)
VIGNETTE_CHARACTERISATION_METHODS.__doc__ = """
Supported vignette characterisation methods.
"""


def characterise_vignette(
    image: ArrayLike,
    method: Union[
        Literal["2D Function", "Bivariate Spline", "RBF"], str
    ] = "RBF",
    **kwargs,
) -> DataVignetteCharacterisation:
    """
    Characterise the vignette of given image using given method.

    Parameters
    ----------
    image
        Image to characterise the vignette of.
    method
        Vignette characterisation method.

    Other Parameters
    ----------------
    function
        {:func:`colour_hdri.distortion.characterise_vignette_2D_function`},
        Characterisation function.
    pre_denoise_sigma
        {:func:`colour_hdri.distortion.characterise_vignette_bivariate_spline`},
        Standard deviation of the gaussian filtering kernel applied on the
        image.
    post_denoise_sigma
        {:func:`colour_hdri.distortion.characterise_vignette_bivariate_spline`},
        Standard deviation of the gaussian filtering kernel applied on the
        resampled image at given ``samples`` count.
    samples
        {:func:`colour_hdri.distortion.characterise_vignette_bivariate_spline`},
        Samples count of the resampled image on the long edge.
    degree
        {:func:`colour_hdri.distortion.characterise_vignette_bivariate_spline`},
        Degree of the bivariate spline.
    denoise_sigma
        {:func:`colour_hdri.distortion.characterise_vignette_RBF`},
        Standard deviation of the gaussian filtering kernel applied on the
        image.

    Returns
    -------
    :class:`DataVignetteCharacterisation`
        Vignette characterisation.

    Examples
    --------
    >>> image = apply_radial_gradient(np.ones([300, 400]))
    >>> parameters, principal_point = characterise_vignette(image).values
    >>> parameters.shape
    (180, 1)
    >>> principal_point  # doctest: +ELLIPSIS
    array([ 0.4983333...,  0.49875   ])
    >>> parameters, principal_point = characterise_vignette(
    ...     image, method="RBF"
    ... ).values
    >>> parameters.shape
    (180, 1)
    >>> principal_point  # doctest: +ELLIPSIS
    array([ 0.4983333...,  0.49875   ])
    >>> parameters, principal_point = characterise_vignette(
    ...     image, method="2D Function"
    ... ).values
    >>> parameters.shape
    (1, 6)
    >>> principal_point  # doctest: +ELLIPSIS
    array([ 0.4983333...,  0.49875   ])
    >>> parameters, principal_point = characterise_vignette(
    ...     image, method="Bivariate Spline"
    ... ).values
    >>> parameters.shape
    (37, 50, 1)
    >>> principal_point  # doctest: +ELLIPSIS
    array([ 0.4983333...,  0.49875   ])
    """

    method = validate_method(method, VIGNETTE_CHARACTERISATION_METHODS)

    return VIGNETTE_CHARACTERISATION_METHODS[method](image, **kwargs)


VIGNETTE_CORRECTION_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "2D Function": correct_vignette_2D_function,
        "Bivariate Spline": correct_vignette_bivariate_spline,
        "RBF": correct_vignette_RBF,
    }
)
VIGNETTE_CHARACTERISATION_METHODS.__doc__ = """
Supported vignette correction methods.
"""


def correct_vignette(
    image: ArrayLike,
    characterisation_data: DataVignetteCharacterisation,
    method: Union[
        Literal["2D Function", "Bivariate Spline", "RBF"], str
    ] = "RBF",
    **kwargs,
) -> NDArray:
    """
    Correct the vignette of given image using given method.

    Parameters
    ----------
    image
        Image to correct the vignette of.
    characterisation_data
         Vignette characterisation data for given function.
    method
        Vignette characterisation method.

    Other Parameters
    ----------------
    function
        {:func:`colour_hdri.distortion.correct_vignette_2D_function`},
        Characterisation function.
    degree
        {:func:`colour_hdri.distortion.correct_vignette_bivariate_spline`},
        Degree of the bivariate spline.
    smoothing
        {:func:`colour_hdri.distortion.correct_vignette_RBF`},
        Smoothing parameter, see :class:`scipy.interpolate.RBFInterpolator`
        class.
    kernel
        {:func:`colour_hdri.distortion.correct_vignette_RBF`},
         Type of RBF, see :class:`scipy.interpolate.RBFInterpolator` class.
    epsilon
        {:func:`colour_hdri.distortion.correct_vignette_RBF`},
        Shape parameter that scales the input to the RBF, see
        :class:`scipy.interpolate.RBFInterpolator` class.

    Returns
    -------
    :class:`numpy.ndarray`
        Vignette corrected image.

    Examples
    --------
    >>> image = apply_radial_gradient(np.ones([5, 7]))
    >>> characterisation_data = characterise_vignette(image)
    >>> np.around(correct_vignette_RBF(image, characterisation_data), 3)
    array([[ 0.   ,  0.091,  0.841,  1.134,  0.841,  0.091,  0.   ],
           [ 0.   ,  0.967,  2.03 ,  2.552,  2.03 ,  0.967,  0.   ],
           [ 0.   ,  1.323,  2.647,  3.97 ,  2.647,  1.323,  0.   ],
           [ 0.   ,  0.967,  2.03 ,  2.552,  2.03 ,  0.967,  0.   ],
           [ 0.   ,  0.091,  0.841,  1.134,  0.841,  0.091,  0.   ]])
    >>> characterisation_data = characterise_vignette(image, method="RBF")
    >>> np.around(
    ...     correct_vignette(image, characterisation_data, method="RBF"), 3
    ... )
    array([[ 0.   ,  0.091,  0.841,  1.134,  0.841,  0.091,  0.   ],
           [ 0.   ,  0.967,  2.03 ,  2.552,  2.03 ,  0.967,  0.   ],
           [ 0.   ,  1.323,  2.647,  3.97 ,  2.647,  1.323,  0.   ],
           [ 0.   ,  0.967,  2.03 ,  2.552,  2.03 ,  0.967,  0.   ],
           [ 0.   ,  0.091,  0.841,  1.134,  0.841,  0.091,  0.   ]])
    >>> characterisation_data = characterise_vignette(
    ...     image, method="2D Function"
    ... )
    >>> np.around(
    ...     correct_vignette(
    ...         image, characterisation_data, method="2D Function"
    ...     ),
    ...     3,
    ... )
    array([[-0.   ,  0.122,  0.597,  0.747,  0.781,  1.08 , -0.   ],
           [ 0.   ,  0.413,  0.676,  0.82 ,  0.76 ,  0.576,  0.   ],
           [ 0.   ,  0.468,  0.759,  1.103,  0.838,  0.611,  0.   ],
           [ 0.   ,  0.439,  0.709,  0.858,  0.801,  0.628, -0.   ],
           [-0.   ,  0.193,  0.742,  0.913,  1.049, -0.477, -0.   ]])
    >>> characterisation_data = characterise_vignette(
    ...     image, method="Bivariate Spline"
    ... )
    >>> np.around(
    ...     correct_vignette(
    ...         image, characterisation_data, method="Bivariate Spline"
    ...     ),
    ...     3,
    ... )
    array([[  0.   ,   0.345,   3.059,   4.072,   3.059,   0.345,   0.   ],
           [  0.   ,   3.624,   7.304,   9.058,   7.304,   3.624,   0.   ],
           [  0.   ,   4.936,   9.481,  14.032,   9.481,   4.936,   0.   ],
           [  0.   ,   3.624,   7.304,   9.058,   7.304,   3.624,   0.   ],
           [  0.   ,   0.345,   3.059,   4.072,   3.059,   0.345,   0.   ]])
    """

    method = validate_method(method, VIGNETTE_CORRECTION_METHODS)

    return VIGNETTE_CORRECTION_METHODS[method](
        image, characterisation_data, **kwargs
    )
