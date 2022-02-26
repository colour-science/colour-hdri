"""
Grossberg (2003) Histogram Based Image Sampling
===============================================

Defines the *Grossberg (2003)* histogram based image sampling objects:

-   :func:`colour_hdri.samples_Grossberg2003`

References
----------
-   :cite:`Banterle2014a` : Banterle, F., & Benedetti, L. (2014). PICCANTE: An
    Open and Portable Library for HDR Imaging.
-   :cite:`Grossberg2003g` : Grossberg, M. D., & Nayar, S. K. (2003).
    Determining the camera response from images: What is knowable? IEEE
    Transactions on Pattern Analysis and Machine Intelligence, 25(11),
    1455-1467. doi:10.1109/TPAMI.2003.1240119
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, Integer, NDArray
from colour.utilities import as_float_array, tsplit, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "samples_Grossberg2003",
]


def samples_Grossberg2003(
    image_stack: ArrayLike, samples: Integer = 1000, n: Integer = 256
) -> NDArray:
    """
    Return the samples for given image stack intensity histograms using
    *Grossberg (2003)* method.

    Parameters
    ----------
    image_stack
        Stack of single channel or multi-channel floating point images.
    samples
        Samples count.
    n
        Histograms bins count.

    Returns
    -------
    :class:`numpy.ndarray`
        Intensity histograms samples.

    References
    ----------
    :cite:`Banterle2014a`, :cite:`Grossberg2003g`
    """

    image_stack = as_float_array(image_stack)

    if image_stack.ndim == 3:
        channels_c = 1
    else:
        channels_c = image_stack.shape[-2]

    cdf_i = []
    for image in tsplit(image_stack):
        histograms = tstack(
            [
                np.histogram(image[..., c], n, range=(0, 1))[0]
                for c in np.arange(channels_c)
            ]
        )
        cdf = np.cumsum(histograms, axis=0)
        cdf_i.append(cdf.astype(np.float_) / np.max(cdf, axis=0))

    samples_cdf_i = np.zeros((samples, len(cdf_i), channels_c))
    samples_u = np.linspace(0, 1, samples)
    for i in np.arange(samples):
        for j in np.arange(channels_c):
            for k, cdf in enumerate(cdf_i):
                samples_cdf_i[i, k, j] = np.argmin(
                    np.abs(cdf[:, j] - samples_u[i])
                )

    return samples_cdf_i
