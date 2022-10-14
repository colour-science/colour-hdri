"""
HDRI Generation
===============

Defines the HDRI generation objects:

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

import numpy as np

from colour.utilities import as_float_array, tsplit, tstack, warning
from colour.hints import ArrayLike, Callable, NDArray, Optional

from colour_hdri.exposure import average_luminance
from colour_hdri.generation import weighting_function_Debevec1997
from colour_hdri.utilities import ImageStack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "image_stack_to_HDRI",
]


def image_stack_to_HDRI(
    image_stack: ImageStack,
    weighting_function: Callable = weighting_function_Debevec1997,
    camera_response_functions: Optional[ArrayLike] = None,
) -> Optional[NDArray]:
    """
    Generate a HDRI from given image stack.

    Parameters
    ----------
    image_stack
        Stack of single channel or multi-channel floating point images. The
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
    thus recommended encoding the images in a wider RGB colourspace or clamp
    negative values.

    References
    ----------
    :cite:`Banterle2011n`
    """

    image_c: Optional[NDArray] = None
    weight_c: Optional[NDArray] = None
    for i, image in enumerate(image_stack):
        if image.data is not None and image.metadata is not None:
            if image_c is None:
                image_c = np.zeros(image.data.shape)
                weight_c = np.zeros(image.data.shape)

            L = 1 / average_luminance(
                image.metadata.f_number,
                image.metadata.exposure_time,
                image.metadata.iso,
            )

            if np.any(image.data <= 0):
                warning(
                    f'"{image.path}" image channels contain negative or equal '
                    f"to zero values, unpredictable results may occur! Please "
                    f"consider encoding your images in a wider gamut RGB "
                    f"colourspace or clamp negative values."
                )

            weights = weighting_function(image.data)
            if i == 0:
                weights[image.data >= 0.5] = 1
            if i == len(image_stack) - 1:
                weights[image.data <= 0.5] = 1

            image_data = image.data
            if camera_response_functions is not None:
                camera_response_functions = as_float_array(
                    camera_response_functions
                )
                samples = np.linspace(0, 1, camera_response_functions.shape[0])

                R, G, B = tsplit(image.data)
                R = np.interp(R, samples, camera_response_functions[..., 0])
                G = np.interp(G, samples, camera_response_functions[..., 1])
                B = np.interp(B, samples, camera_response_functions[..., 2])
                image_data = tstack([R, G, B])

            image_c += weights * image_data / L
            weight_c += weights

    if image_c is not None and weight_c is not None:
        image_c /= weight_c

    return image_c
