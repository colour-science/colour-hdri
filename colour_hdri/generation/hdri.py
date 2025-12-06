"""
HDRI Generation
===============

Define the HDRI generation objects:

-   :func:`colour_hdri.image_stack_to_HDRI`

See Also
--------
`Colour - HDRI - Examples Jupyter Notebooks
<https://github.com/colour-science/colour-hdri/\
blob/master/colour_hdri/examples>`__

References
----------
-   :cite:`Banterle2011n` : Banterle, F., Artusi, A., Debattista, K., &
    Chalmers, A. (2011). 2.1.1 Generating HDR Content by Combining Multiple
    Exposures. In Advanced High Dynamic Range Imaging. A K Peters/CRC Press.
    ISBN:978-1-56881-719-4
"""

from __future__ import annotations

import gc
import typing

import numpy as np
from colour.constants import EPSILON

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Callable, NDArrayFloat

from colour.utilities import as_float_array, attest, tsplit, tstack, warning, zeros

from colour_hdri.exposure import average_luminance
from colour_hdri.generation import weighting_function_Debevec1997

if typing.TYPE_CHECKING:
    from colour_hdri.utilities import ImageStack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "image_stack_to_HDRI",
]


def image_stack_to_HDRI(
    image_stack: ImageStack,
    weighting_function: Callable = weighting_function_Debevec1997,
    camera_response_functions: ArrayLike | None = None,
) -> NDArrayFloat:
    """
    Generate a HDRI from given image stack.

    Parameters
    ----------
    image_stack
        Stack of single channel or multichannel floating point images. The
        stack is assumed to be representing linear values except if
        ``camera_response_functions`` argument is provided.
    weighting_function
        Weighting function :math:`w`.
    camera_response_functions
        Camera response functions :math:`g(z)` of the imaging system / camera
        if the stack is representing non-linear values.

    Returns
    -------
    :class:`numpy.ndarray`
        HDRI.

    Warnings
    --------
    If the image stack contains images with negative or equal to zero values,
    unpredictable results may occur and NaNs might be generated. It is
    thus recommended encoding the images in a wider RGB colourspace. This
    definition avoids NaNs creation by ensuring that all values are greater or
    equal to current floating point format epsilon. In practical applications
    such as HDRI merging with photographic material there should never be a
    pixel with a value exactly equal to zero. Ideally, the process should not
    be presented by any negative photometric quantity  even though RGB
    colourspace encodings allows to do so.

    References
    ----------
    :cite:`Banterle2011n`
    """

    attest(len(image_stack) > 0, "Image stack cannot be empty!")

    attest(image_stack.is_valid(), "Image stack is invalid!")

    image_c = as_float_array([])
    weight_c = as_float_array([])

    for i, image in enumerate(image_stack):
        if image.data is None:
            image.read_data(image_stack.cctf_decoding)

        if image_c.size == 0:
            image_c = zeros(image.data.shape)  # pyright: ignore
            weight_c = zeros(image.data.shape)  # pyright: ignore

        L = 1 / average_luminance(
            image.metadata.f_number,  # pyright: ignore
            image.metadata.exposure_time,  # pyright: ignore
            image.metadata.iso,  # pyright: ignore
        )

        if np.any(image.data <= 0):  # pyright: ignore
            warning(
                f'"{image.path}" image channels contain negative or equal '
                f"to zero values, unpredictable results may occur! Please "
                f"consider encoding your images in a wider gamut RGB "
                f"colourspace."
            )

        data = np.clip(image.data, EPSILON, 1)  # pyright: ignore
        weights = np.clip(weighting_function(data), EPSILON, 1)

        # Invoking garbage collection to free memory.
        image.data = None
        gc.collect()

        if i == 0:
            weights[data >= 0.5] = 1

        if i == len(image_stack) - 1:
            weights[data <= 0.5] = 1

        if camera_response_functions is not None:
            camera_response_functions = as_float_array(camera_response_functions)
            samples = np.linspace(0, 1, camera_response_functions.shape[0])

            R, G, B = tsplit(data)
            R = np.interp(R, samples, camera_response_functions[..., 0])
            G = np.interp(G, samples, camera_response_functions[..., 1])
            B = np.interp(B, samples, camera_response_functions[..., 2])
            data = tstack([R, G, B])

        image_c += weights * data / L
        weight_c += weights

        del data, weights

    return image_c / weight_c
