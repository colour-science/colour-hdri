"""
Lens Vignette Characterisation & Correction
===========================================

Defines various objects to correct camera lens vignette:

-   :func:`colour_hdri.characterise_vignette_radial`
-   :func:`colour_hdri.correct_vignette_radial`
"""

from __future__ import annotations

import numpy as np

from colour.hints import (
    ArrayLike,
    Floating,
    FloatingOrNDArray,
    NDArray,
    Tuple,
    cast,
)
from colour.utilities import (
    as_int_array,
    as_float_array,
    full,
    orient,
    tsplit,
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
    "apply_gradient_radial",
    "symmetrise_image",
    "extend_image",
    "function_vignette_radial",
    "vignette_center_radial",
    "vignette_coefficients_radial",
    "characterise_vignette_radial",
    "correct_vignette_radial",
]


def apply_gradient_radial(
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
    >>> np.around(apply_gradient_radial(np.ones([5, 7])), 3)
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


def symmetrise_image(image: ArrayLike, center: ArrayLike) -> NDArray:
    """
    Generate a symmetrical image by duplicating the largest image quadrant
    bounded by given center.
    """

    image = as_float_array(image)
    center = as_float_array(center)

    shape_x, shape_y = image.shape[0:2]
    half_x, half_y = shape_x // 2, shape_y // 2
    center_x, center_y = as_int_array(
        center * as_float_array([shape_x, shape_y])
    )

    if center_x < half_x:
        if center_y < half_y:
            quadrant = image[center_x:, center_y:, ...]
            quadrant = np.concatenate(
                [cast(NDArray, orient(quadrant, "flop")), quadrant], axis=0
            )
            quadrant = np.concatenate(
                [cast(NDArray, orient(quadrant, "flip")), quadrant], axis=1
            )
        else:
            quadrant = image[center_x:, 0:center_y, ...]
            quadrant = np.concatenate(
                [cast(NDArray, orient(quadrant, "flop")), quadrant], axis=0
            )
            quadrant = np.concatenate(
                [quadrant, cast(NDArray, orient(quadrant, "flip"))], axis=1
            )
    else:
        if center_y < half_y:
            quadrant = image[0:center_x, center_y:, ...]
            quadrant = np.concatenate(
                [quadrant, cast(NDArray, orient(quadrant, "flop"))], axis=0
            )
            quadrant = np.concatenate(
                [cast(NDArray, orient(quadrant, "flip")), quadrant], axis=1
            )
        else:
            quadrant = image[0:center_x, 0:center_y, ...]
            quadrant = np.concatenate(
                [quadrant, cast(NDArray, orient(quadrant, "flop"))], axis=0
            )
            quadrant = np.concatenate(
                [quadrant, cast(NDArray, orient(quadrant, "flip"))], axis=1
            )

    return quadrant


def extend_image(
    image: ArrayLike, center: ArrayLike, fill_value: Floating = 0.18
) -> Tuple[NDArray, Tuple[slice, slice]]:
    """
    Pass
    """

    image = as_float_array(image)
    center = as_float_array(center)

    shape_x, shape_y = image.shape[0:2]
    half_x, half_y = shape_x // 2, shape_y // 2
    center_x, center_y = as_int_array(
        center * as_float_array([shape_x, shape_y])
    )

    if center_x < half_x:
        if center_y < half_y:
            padded_image = full(
                (
                    (half_x - center_x + half_x) * 2,
                    (half_y - center_y + half_y) * 2,
                    3,
                ),
                fill_value,
            )
            crop_x = slice(
                half_x + half_x - center_x - center_x, padded_image.shape[0]
            )
            crop_y = slice(
                half_y + half_y - center_y - center_y, padded_image.shape[1]
            )
        else:
            padded_image = full(
                ((half_x - center_x + half_x) * 2, center_y * 2, 3), fill_value
            )
            crop_x = slice(
                half_x + half_x - center_x - center_x, padded_image.shape[0]
            )
            crop_y = slice(0, shape_y)
    else:
        if center_y < half_y:
            padded_image = full(
                (center_x * 2, (half_y - center_y + half_y) * 2, 3), fill_value
            )
            crop_x = slice(0, shape_x)
            crop_y = slice(
                half_y + half_y - center_y - center_y, padded_image.shape[1]
            )
        else:
            padded_image = full((center_x * 2, center_y * 2, 3), fill_value)
            crop_x = slice(0, shape_x)
            crop_y = slice(0, shape_y)

    padded_image[crop_x, crop_y, ...] = image

    return padded_image, (crop_x, crop_y)


def function_vignette_radial(
    x: FloatingOrNDArray,
    a: FloatingOrNDArray,
    b: FloatingOrNDArray,
    c: FloatingOrNDArray,
    d: FloatingOrNDArray,
    e: FloatingOrNDArray,
) -> FloatingOrNDArray:
    """
    Pass
    """

    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def vignette_center_radial(
    image: ArrayLike, threshold: Floating = 0.99
) -> NDArray:
    """
    Pass
    """

    image = as_float_array(image)
    shape_x, shape_y, _ = image.shape

    L = np.median(image, axis=-1)

    center = zeros(L.shape)
    center[L > np.max(L) * threshold] = 1

    return center_of_mass(center) / as_float_array([shape_x, shape_y])


def vignette_coefficients_radial(image: ArrayLike) -> NDArray:
    """
    Pass
    """

    image = np.atleast_3d(image)

    width, height, channels = image.shape
    ratio = height / width

    samples_x = np.linspace(-1 * ratio, 1 * ratio, height)
    samples_y = np.linspace(-1, 1, width)
    distance = np.sqrt(
        (samples_x**2)[np.newaxis, ...] + (samples_y**2)[..., np.newaxis]
    )

    import matplotlib.pyplot as plt

    colour.plotting.colour_style()
    colour.plotting.plot_image(image)

    samples = np.linspace(np.min(distance), np.max(distance), distance.size)
    coefficients = []
    for i in range(channels):
        coefficients.append(
            curve_fit(
                function_vignette_radial,
                np.ravel(distance),
                np.ravel(image[..., i]),
            )[0]
        )

        plt.scatter(distance, image[..., i], s=5, color="r", alpha=0.15)
        plt.plot(samples, function_vignette_radial(samples, *coefficients[i]))

    return as_float_array(coefficients)


def characterise_vignette_radial(
    image: ArrayLike,
) -> Tuple[NDArray, NDArray]:
    """
    Pass
    """

    image = np.nan_to_num(image)

    center = vignette_center_radial(image)
    symmetrised_image = symmetrise_image(image, center)

    coefficients = vignette_coefficients_radial(symmetrised_image)

    return coefficients, center


def correct_vignette_radial(
    image: ArrayLike, coefficients: ArrayLike, center: ArrayLike
) -> NDArray:
    """
    Pass
    """

    image = np.atleast_3d(image)
    coefficients = as_float_array(coefficients)

    padded, (crop_x, crop_y) = extend_image(image, center)

    width, height, channels = padded.shape
    ratio = height / width

    samples_x = np.linspace(-1 * ratio, 1 * ratio, height)
    samples_y = np.linspace(-1, 1, width)
    distance = np.sqrt(
        (samples_x**2)[np.newaxis, ...] + (samples_y**2)[..., np.newaxis]
    )

    for i in range(channels):
        padded[..., i] = padded[..., i] / function_vignette_radial(
            distance, *coefficients[i]
        )

    return np.squeeze(padded[crop_x, crop_y])


if __name__ == "__main__":
    import colour

    WIDTH, HEIGHT = 200, 300

    IMAGE = colour.utilities.ones([HEIGHT, WIDTH, 3])
    IMAGE_VIGNETTE = np.copy(IMAGE)

    IMAGE_VIGNETTE = apply_gradient_radial(
        IMAGE, (2, 2), (0.5, 0.5), 1, 1, 0.1
    )
    colour.plotting.plot_image(IMAGE_VIGNETTE)

    # IMAGE_VIGNETTE[..., 0] = apply_gradient_radial(
    #     IMAGE[..., 0], (1, 1), (0.25, -0.25), 0.5, 1.5, 0.15
    # )
    # IMAGE_VIGNETTE[..., 1] = apply_gradient_radial(
    #     IMAGE[..., 1], (1, 1), (0.25, -0.25), 0.5, 1.55, 0.15
    # )
    # IMAGE_VIGNETTE[..., 2] = apply_gradient_radial(
    #     IMAGE[..., 2], (1, 1), (0.25, -0.25), 0.5, 1.6, 0.15
    # )

    # CENTER = vignette_center_radial(IMAGE_VIGNETTE)
    # CENTER_X, CENTER_Y = as_int_array(CENTER * as_float_array([HEIGHT, WIDTH]))
    #
    # IMAGE_VIGNETTE_CENTER = np.copy(IMAGE_VIGNETTE)
    # IMAGE_VIGNETTE_CENTER[CENTER_X, CENTER_Y] = 0
    # # IMAGE_VIGNETTE_CENTER[128, 384 // 2] = 0
    # colour.plotting.plot_image(IMAGE_VIGNETTE_CENTER)
    # IMAGE_VIGNETTE_SYMMETRICAL = symmetrise_image(
    #     IMAGE_VIGNETTE_CENTER, CENTER
    # )
    #
    # COEFFICIENTS = vignette_coefficients_radial(IMAGE_VIGNETTE_SYMMETRICAL)
    #
    # PADDED_IMAGE, (crop_x, crop_y) = extend_image(
    #     IMAGE_VIGNETTE_CENTER, CENTER
    # )
    # colour.plotting.plot_image(PADDED_IMAGE[crop_x, crop_y])

    CORRECTED = correct_vignette_radial(
        IMAGE_VIGNETTE, *characterise_vignette_radial(IMAGE)
    )

    print(np.min(CORRECTED), np.max(CORRECTED))

    colour.plotting.plot_image(CORRECTED)
