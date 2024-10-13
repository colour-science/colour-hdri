"""
Clipped Highlights Recovery
===========================

Define the clipped highlights recovery objects:

-   :func:`colour_hdri.highlights_recovery_blend`
-   :func:`colour_hdri.highlights_recovery_LCHab`

See Also
--------
`Colour - HDRI - Examples: Merge from Raw Files Jupyter Notebook
<https://github.com/colour-science/colour-hdri/\
blob/master/colour_hdri/examples/examples_merge_from_raw_files.ipynb>`__

References
----------
-   :cite:`Coffin2015a` : Coffin, D. (2015). dcraw.
    https://www.cybercom.net/~dcoffin/dcraw/
"""

from __future__ import annotations

import numpy as np
from colour.algebra import vecmul
from colour.hints import ArrayLike, NDArrayFloat
from colour.models import Lab_to_LCHab  # pyright: ignore
from colour.models import LCHab_to_Lab  # pyright: ignore
from colour.models import (
    Lab_to_XYZ,
    RGB_Colourspace,
    RGB_COLOURSPACE_sRGB,
    RGB_to_XYZ,
    XYZ_to_Lab,
    XYZ_to_RGB,
)
from colour.utilities import tsplit, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "highlights_recovery_blend",
    "highlights_recovery_LCHab",
]


def highlights_recovery_blend(
    RGB: ArrayLike, multipliers: ArrayLike, threshold: float = 0.99
) -> NDArrayFloat:
    """
    Perform highlights recovery using *Coffin (1997)* method from *dcraw*.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    multipliers
        Normalised camera white level or white balance multipliers.
    threshold
        Threshold for highlights selection.

    Returns
    -------
    :class:`numpy.ndarray`
         Highlights recovered *RGB* colourspace array.

    References
    ----------
    :cite:`Coffin2015a`
    """

    M = np.array(
        [
            [1.0000000, 1.0000000, 1.0000000],
            [1.7320508, -1.7320508, 0.0000000],
            [-1.0000000, -1.0000000, 2.0000000],
        ]
    )

    clipping_level = np.min(multipliers) * threshold

    Lab = vecmul(M, RGB)

    Lab_c = vecmul(M, np.minimum(RGB, clipping_level))

    s = np.sum((Lab * Lab)[..., 1:3], axis=2)
    s_c = np.sum((Lab_c * Lab_c)[..., 1:3], axis=2)

    ratio = np.sqrt(s_c / s)
    ratio[np.logical_or(np.isnan(ratio), np.isinf(ratio))] = 1

    Lab[:, :, 1:3] *= np.rollaxis(ratio[None], 0, 3)

    RGB_o = vecmul(np.linalg.inv(M), Lab)

    return RGB_o


def highlights_recovery_LCHab(
    RGB: ArrayLike,
    threshold: float | None = None,
    RGB_colourspace: RGB_Colourspace = RGB_COLOURSPACE_sRGB,
) -> NDArrayFloat:
    """
    Perform highlights recovery in *CIE L\\*C\\*Hab* colourspace.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    threshold
        Threshold for highlights selection, automatically computed
        if not given.
    RGB_colourspace
        Working *RGB* colourspace to perform the *CIE L\\*C\\*Hab* to and from.

    Returns
    -------
    :class:`numpy.ndarray`
         Highlights recovered *RGB* colourspace array.
    """

    L, _C, H = tsplit(
        Lab_to_LCHab(
            XYZ_to_Lab(
                RGB_to_XYZ(RGB, RGB_colourspace),
                RGB_colourspace.whitepoint,
            )
        )
    )

    _L_c, C_c, _H_c = tsplit(
        Lab_to_LCHab(
            XYZ_to_Lab(
                RGB_to_XYZ(
                    np.clip(RGB, 0, threshold),
                    RGB_colourspace,
                ),
                RGB_colourspace.whitepoint,
            )
        )
    )

    return XYZ_to_RGB(
        Lab_to_XYZ(LCHab_to_Lab(tstack([L, C_c, H])), RGB_colourspace.whitepoint),
        RGB_colourspace,
    )
