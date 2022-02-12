"""
Tonemapping Operators Plotting
==============================

Defines the tonemapping operators plotting objects:

-   :func:`colour_hdri.plotting.plot_tonemapping_operator_image`
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from colour.hints import Any, ArrayLike, Boolean, Callable, Dict, Tuple
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    artist,
    override_style,
    render,
)
from colour.utilities import as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_tonemapping_operator_image",
]


@override_style()
def plot_tonemapping_operator_image(
    image: ArrayLike,
    luminance_function: ArrayLike,
    log_scale: Boolean = False,
    cctf_encoding: Callable = CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given tonemapped image with superimposed luminance mapping function.

    Parameters
    ----------
    image
         Tonemapped image to plot.
    luminance_function
        Luminance mapping function.
    log_scale
        Use a log scale for plotting the luminance mapping function.
    cctf_encoding
        Encoding colour component transfer function / opto-electronic
        transfer function used for plotting.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.
    """

    image = as_float_array(image)
    luminance_function = as_float_array(luminance_function)

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    shape = image.shape
    bounding_box = [0, 1, 0, 1]

    image = np.clip(cctf_encoding(image), 0, 1)
    axes.imshow(
        image,
        aspect=shape[0] / shape[1],
        extent=bounding_box,
        interpolation="nearest",
    )

    axes.plot(
        np.linspace(0, 1, len(luminance_function)),
        luminance_function,
        color="red",
    )

    settings = {
        "axes": axes,
        "bounding_box": bounding_box,
        "x_ticker": True,
        "y_ticker": True,
        "x_label": "Input Luminance",
        "y_label": "Output Luminance",
    }
    settings.update(kwargs)

    if log_scale:
        settings.update(
            {
                "x_label": "$log_2$ Input Luminance",
                "x_ticker_locator": matplotlib.ticker.AutoMinorLocator(0.5),
            }
        )
        plt.gca().set_xscale("log", basex=2)
        plt.gca().xaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )

    return render(**settings)
