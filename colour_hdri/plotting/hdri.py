"""
HDRI Plotting
=============

Define the HDRI plotting objects:

-   :func:`colour_hdri.plotting.plot_HDRI_strip`
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colour.hints import Any, ArrayLike, Callable, Tuple
from colour.plotting import CONSTANTS_COLOUR_STYLE, override_style, render
from colour.utilities import as_float_array
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour_hdri.exposure import adjust_exposure

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_HDRI_strip",
]


@override_style()
def plot_HDRI_strip(
    image: ArrayLike,
    count: int = 5,
    ev_steps: float = -2,
    cctf_encoding: Callable = CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given HDRI as strip of images of varying exposure.

    Parameters
    ----------
    image
         HDRI to plot.
    count
        Strip images count.
    ev_steps
        Exposure variation for each image of the strip.
    cctf_encoding
        Encoding colour component transfer function / opto-electronic
        transfer function used for plotting.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.display`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.
    """

    image = as_float_array(image)

    grid = mpl.gridspec.GridSpec(1, count)  # pyright: ignore
    grid.update(wspace=0, hspace=0)

    height, width, _channel = image.shape
    for i in range(count):
        ev = i * ev_steps
        axis = plt.subplot(grid[i])
        axis.imshow(np.clip(cctf_encoding(adjust_exposure(image, ev)), 0, 1))
        axis.text(width * 0.05, height - height * 0.05, f"EV {ev}", color=(1, 1, 1))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")

    return render(**kwargs)
