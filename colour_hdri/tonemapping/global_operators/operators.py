"""
Global Tonemapping Operators
============================

Defines the global tonemapping operators objects:

-   :func:`colour_hdri.tonemapping_operator_simple`
-   :func:`colour_hdri.tonemapping_operator_normalisation`
-   :func:`colour_hdri.tonemapping_operator_gamma`
-   :func:`colour_hdri.tonemapping_operator_logarithmic`
-   :func:`colour_hdri.tonemapping_operator_exponential`
-   :func:`colour_hdri.tonemapping_operator_logarithmic_mapping`
-   :func:`colour_hdri.tonemapping_operator_exponentiation_mapping`
-   :func:`colour_hdri.tonemapping_operator_Schlick1994`
-   :func:`colour_hdri.tonemapping_operator_Tumblin1999`
-   :func:`colour_hdri.tonemapping_operator_Reinhard2004`
-   :func:`colour_hdri.tonemapping_operator_filmic`

See Also
--------
`Colour - HDRI - Examples: Global Tonemapping Operators Jupyter Notebook
<https://github.com/colour-science/colour-hdri/blob/master/colour_hdri/\

examples/examples_global_tonemapping_operators.ipynb>`__

References
----------
-   :cite:`Banterle2011k` : Banterle, F., Artusi, A., Debattista, K., &
    Chalmers, A. (2011). 3.2.1 Simple Mapping Methods. In Advanced High
    Dynamic Range Imaging (pp. 38-41). A K Peters/CRC Press.
    ISBN:978-1-56881-719-4
-   :cite:`Habble2010d` : Habble, J. (2010). Filmic Tonemapping Operators.
    Retrieved March 15, 2015, from http://filmicgames.com/archives/75
-   :cite:`Habble2010e` : Habble, J. (2010). Uncharted 2: HDR Lighting.
    Retrieved March 15, 2015, from
    http://www.slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting
-   :cite:`Reinhard2005c` : Reinhard, E., & Devlin, K. (2005). Dynamic Range
    Reduction Inspired by Photoreceptor Physiology. IEEE Transactions on
    Visualization and Computer Graphics, 11(01), 13-24. doi:10.1109/TVCG.2005.9
-   :cite:`Schlick1994` : Schlick, C. (1994). Quantization Techniques for
    Visualization of High Dynamic Range Pictures. Proceedings of the Fifth
    Eurographics Workshop on Rendering, Section 5, 7-18. ISSN:0920-5691
-   :cite:`Tumblin1999c` : Tumblin, J., Hodgins, J. K., & Guenter, B. K.
    (1999). Two methods for display of high contrast images. ACM Transactions
    on Graphics, 18(1), 56-94. doi:10.1145/300776.300783
-   :cite:`Wikipediabn` : Wikipedia. (n.d.). Tonemapping - Purpose and
    methods. Retrieved March 15, 2015, from
    http://en.wikipedia.org/wiki/Tone_mapping#Purpose_and_methods
"""

from __future__ import annotations

import numpy as np

from colour.constants import EPSILON
from colour.hints import (
    ArrayLike,
    Floating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
)
from colour.models import RGB_COLOURSPACES, RGB_Colourspace, RGB_luminance
from colour.utilities import as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "log_average",
    "tonemapping_operator_simple",
    "tonemapping_operator_normalisation",
    "tonemapping_operator_gamma",
    "tonemapping_operator_logarithmic",
    "tonemapping_operator_exponential",
    "tonemapping_operator_logarithmic_mapping",
    "tonemapping_operator_exponentiation_mapping",
    "tonemapping_operator_Schlick1994",
    "tonemapping_operator_Tumblin1999",
    "tonemapping_operator_Reinhard2004",
    "tonemapping_operator_filmic",
]


def log_average(
    a: FloatingOrArrayLike, epsilon: Floating = EPSILON
) -> FloatingOrNDArray:
    """
    Compute the log average of given array.

    Parameters
    ----------
    a
        Array to compute the log average.
    epsilon
        Constant to avoid singularities in computations.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Array log average.

    Examples
    --------
    >>> log_average(np.linspace(0, 10, 10))  # doctest: +ELLIPSIS
    0.1...
    """

    a = as_float_array(a)

    average = np.exp(np.average(np.log(a + epsilon)))

    return average


def tonemapping_operator_simple(RGB: ArrayLike) -> NDArray:
    """
    Perform given *RGB* array tonemapping using the simple method:
    :math:`\\cfrac{RGB}{RGB + 1}`.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Wikipediabn`

    Examples
    --------
    >>> tonemapping_operator_simple(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     )
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.3245382...,  0.2601156...,  0.1911532...],
            [ 0.5830618...,  0.3567839...,  0.2808993...]],
    <BLANKLINE>
           [[ 0.8150290...,  0.6831692...,  0.5733340...],
            [ 0.8683127...,  0.7746486...,  0.6893211...]]])
    """

    RGB = as_float_array(RGB)

    return RGB / (RGB + 1)


def tonemapping_operator_normalisation(
    RGB: ArrayLike, colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"]
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using the normalisation method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Banterle2011k`

    Examples
    --------
    >>> tonemapping_operator_normalisation(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     )
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.1194997...,  0.0874388...,  0.0587783...],
            [ 0.3478122...,  0.1379590...,  0.0971544...]],
    <BLANKLINE>
           [[ 1.0959009...,  0.5362936...,  0.3342115...],
            [ 1.6399638...,  0.8549608...,  0.5518382...]]])
    """

    RGB = as_float_array(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = as_float_array(np.max(L))

    return RGB / L_max


def tonemapping_operator_gamma(
    RGB: ArrayLike, gamma: Floating = 1, EV: Floating = 0
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using the gamma and exposure
    correction method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    gamma
        :math:`\\gamma` correction value.
    EV
        Exposure adjustment value.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Banterle2011k`

    Examples
    --------
    >>> tonemapping_operator_gamma(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     ),
    ...     1.0,
    ...     -3.0,
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.0600585...,  0.0439453...,  0.0295410...],
            [ 0.1748046...,  0.0693359...,  0.0488282...]],
    <BLANKLINE>
           [[ 0.5507817...,  0.2695323...,  0.1679692...],
            [ 0.8242187...,  0.4296892...,  0.2773447...]]])
    """

    RGB = as_float_array(RGB)

    exposure = 2**EV
    RGB = (exposure * RGB) ** (1 / gamma)

    return RGB


def tonemapping_operator_logarithmic(
    RGB: ArrayLike,
    q: Floating = 1,
    k: Floating = 1,
    colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"],
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using the logarithmic method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    q
        :math:`q`.
    k
        :math:`k`.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Banterle2011k`

    Examples
    --------
    >>> tonemapping_operator_logarithmic(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     ),
    ...     1.0,
    ...     25,
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.0884587...,  0.0647259...,  0.0435102...],
            [ 0.2278222...,  0.0903652...,  0.0636376...]],
    <BLANKLINE>
           [[ 0.4717487...,  0.2308565...,  0.1438669...],
            [ 0.5727396...,  0.2985858...,  0.1927235...]]])
    """

    RGB = as_float_array(RGB)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = as_float_array(
        RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    )
    L_max = np.max(L)
    L_d = as_float_array(np.log10(1 + L * q) / np.log10(1 + L_max * k))

    return RGB * L_d[..., None] / L[..., None]


def tonemapping_operator_exponential(
    RGB: ArrayLike,
    q: Floating = 1,
    k: Floating = 1,
    colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"],
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using the exponential method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    q
        :math:`q`.
    k
        :math:`k`.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Banterle2011k`

    Examples
    --------
    >>> tonemapping_operator_exponential(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     ),
    ...     1.0,
    ...     25,
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.0148082...,  0.0108353...,  0.0072837...],
            [ 0.0428669...,  0.0170031...,  0.0119740...]],
    <BLANKLINE>
           [[ 0.1312736...,  0.0642404...,  0.0400338...],
            [ 0.1921684...,  0.1001830...,  0.0646635...]]])
    """

    RGB = as_float_array(RGB)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = as_float_array(
        RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    )
    L_a = log_average(L)
    L_d = as_float_array(1 - np.exp(-(L * q) / (L_a * k)))

    return RGB * L_d[..., None] / L[..., None]


def tonemapping_operator_logarithmic_mapping(
    RGB: ArrayLike,
    p: Floating = 1,
    q: Floating = 1,
    colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"],
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using the logarithmic mapping
    method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    p
        :math:`p`.
    q
        :math:`q`.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Schlick1994`

    Examples
    --------
    >>> tonemapping_operator_logarithmic_mapping(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     )
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.2532899...,  0.1853341...,  0.1245857...],
            [ 0.6523387...,  0.2587489...,  0.1822179...]],
    <BLANKLINE>
           [[ 1.3507897...,  0.6610269...,  0.4119437...],
            [ 1.6399638...,  0.8549608...,  0.5518382...]]])
    """

    RGB = as_float_array(RGB)

    L = as_float_array(
        RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    )
    L_max = np.max(L)
    L_d = as_float_array(
        (np.log(1 + p * L) / np.log(1 + p * L_max)) ** (1 / q)
    )

    return RGB * L_d[..., None] / L[..., None]


def tonemapping_operator_exponentiation_mapping(
    RGB: ArrayLike,
    p: Floating = 1,
    q: Floating = 1,
    colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"],
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using the exponentiation mapping
    method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    p
        :math:`p`.
    q
        :math:`q`.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Schlick1994`

    Examples
    --------
    >>> tonemapping_operator_exponentiation_mapping(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     )
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.1194997...,  0.0874388...,  0.0587783...],
            [ 0.3478122...,  0.1379590...,  0.0971544...]],
    <BLANKLINE>
           [[ 1.0959009...,  0.5362936...,  0.3342115...],
            [ 1.6399638...,  0.8549608...,  0.5518382...]]])
    """

    RGB = as_float_array(RGB)

    L = as_float_array(
        RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    )
    L_max = np.max(L)
    L_d = as_float_array((L / L_max) ** (p / q))

    return RGB * L_d[..., None] / L[..., None]


def tonemapping_operator_Schlick1994(
    RGB: ArrayLike,
    p: Floating = 1,
    colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"],
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using *Schlick (1994)* method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    p
        :math:`p`.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Banterle2011k`, :cite:`Schlick1994`

    Examples
    --------
    >>> tonemapping_operator_Schlick1994(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     )
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.1194997...,  0.0874388...,  0.0587783...],
            [ 0.3478122...,  0.1379590...,  0.0971544...]],
    <BLANKLINE>
           [[ 1.0959009...,  0.5362936...,  0.3342115...],
            [ 1.6399638...,  0.8549608...,  0.5518382...]]])
    """

    # TODO: Implement automatic *p* and *non-uniform* computations support.

    RGB = as_float_array(RGB)

    L = as_float_array(
        RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    )
    L_max = np.max(L)
    L_d = as_float_array((p * L) / (p * L - L + L_max))

    return RGB * L_d[..., None] / L[..., None]


def tonemapping_operator_Tumblin1999(
    RGB: ArrayLike,
    L_da: Floating = 20,
    C_max: Floating = 100,
    L_max: Floating = 100,
    colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"],
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using
    *Tumblin, Hodgins and Guenter (1999)* method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    L_da
        :math:`L_{da}` display adaptation luminance, a mid-range display value.
    C_max
        :math:`C_{max}` maximum contrast available from the display.
    L_max
        :math:`L_{max}` maximum display luminance.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Tumblin1999c`

    Examples
    --------
    >>> tonemapping_operator_Tumblin1999(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     )
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.0400492...,  0.0293043...,  0.0196990...],
            [ 0.1019768...,  0.0404489...,  0.0284852...]],
    <BLANKLINE>
           [[ 0.2490212...,  0.1218618...,  0.0759427...],
            [ 0.3408366...,  0.1776880...,  0.1146895...]]])
    """

    RGB = as_float_array(RGB)

    L_w = as_float_array(
        RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    )

    def f(x: FloatingOrNDArray) -> FloatingOrNDArray:
        return np.where(
            x > 100, 2.655, 1.855 + 0.4 * np.log10(x + 2.3 * 10**-5)
        )

    L_wa = np.exp(np.mean(np.log(L_w + 2.3 * 10**-5)))
    g_d = f(L_da)
    g_w = f(L_wa)
    g_wd = g_w / (1.855 + 0.4 * np.log(L_da))

    mL_wa = np.sqrt(C_max) ** (g_wd - 1)

    L_d = mL_wa * L_da * (L_w / L_wa) ** (g_w / g_d)

    return (RGB * L_d[..., None] / L_w[..., None]) / L_max


def tonemapping_operator_Reinhard2004(
    RGB: ArrayLike,
    f: Floating = 0,
    m: Floating = 0.3,
    a: Floating = 0,
    c: Floating = 0,
    colourspace: RGB_Colourspace = RGB_COLOURSPACES["sRGB"],
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using *Reinhard and Devlin (2004)*
    method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    f
        :math:`f`.
    m
        :math:`m`.
    a
        :math:`a`.
    c
        :math:`c`.
    colourspace
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Reinhard2005c`

    Examples
    --------
    >>> tonemapping_operator_Reinhard2004(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     ),
    ...     -10,
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.0216792...,  0.0159556...,  0.0107821...],
            [ 0.0605894...,  0.0249445...,  0.0176972...]],
    <BLANKLINE>
           [[ 0.1688972...,  0.0904532...,  0.0583584...],
            [ 0.2331935...,  0.1368456...,  0.0928316...]]])
    """

    RGB = as_float_array(RGB)

    C_av = np.array(
        (
            np.average(RGB[..., 0]),
            np.average(RGB[..., 1]),
            np.average(RGB[..., 2]),
        )
    )

    L = as_float_array(
        RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    )

    L_lav = log_average(L)
    L_min, L_max = np.min(L), np.max(L)

    f = np.exp(-f)

    m = (
        m
        if m > 0
        else (
            0.3
            + 0.7
            * (
                (np.log(L_max) - L_lav)
                / (np.log(L_max) - np.log(L_min)) ** 1.4
            )
        )
    )

    I_l = (c * RGB + (1 - c)) * L[..., None]
    I_g = c * C_av + (1 - c) * L_lav
    I_a = a * I_l + (1 - a) * I_g

    return RGB / (RGB + (f * I_a) ** m)


def tonemapping_operator_filmic(
    RGB: ArrayLike,
    shoulder_strength: Floating = 0.22,
    linear_strength: Floating = 0.3,
    linear_angle: Floating = 0.1,
    toe_strength: Floating = 0.2,
    toe_numerator: Floating = 0.01,
    toe_denominator: Floating = 0.3,
    exposure_bias: Floating = 2,
    linear_whitepoint: Floating = 11.2,
) -> NDArray:
    """
    Perform given *RGB* array tonemapping using *Habble (2010)* method.

    Parameters
    ----------
    RGB
        *RGB* array to perform tonemapping onto.
    shoulder_strength
        Shoulder strength.
    linear_strength
        Linear strength.
    linear_angle
        Linear angle.
    toe_strength
        Toe strength.
    toe_numerator
        Toe numerator.
    toe_denominator
        Toe denominator.
    exposure_bias
        Exposure bias.
    linear_whitepoint
        Linear whitepoint.

    Returns
    -------
    :class:`numpy.ndarray`
        Tonemapped *RGB* array.

    References
    ----------
    :cite:`Habble2010d`, :cite:`Habble2010e`

    Examples
    --------
    >>> tonemapping_operator_filmic(
    ...     np.array(
    ...         [
    ...             [
    ...                 [0.48046875, 0.35156256, 0.23632812],
    ...                 [1.39843753, 0.55468757, 0.39062594],
    ...             ],
    ...             [
    ...                 [4.40625388, 2.15625895, 1.34375372],
    ...                 [6.59375023, 3.43751395, 2.21875829],
    ...             ],
    ...         ]
    ...     )
    ... )  # doctest: +ELLIPSIS
    array([[[ 0.4507954...,  0.3619673...,  0.2617269...],
            [ 0.7567191...,  0.4933310...,  0.3911730...]],
    <BLANKLINE>
           [[ 0.9725554...,  0.8557374...,  0.7465713...],
            [ 1.0158782...,  0.9382937...,  0.8615161...]]])
    """

    RGB = as_float_array(RGB)

    A = shoulder_strength
    B = linear_strength
    C = linear_angle
    D = toe_strength
    E = toe_numerator
    F = toe_denominator

    def f(x: FloatingOrNDArray, A, B, C, D, E, F) -> FloatingOrNDArray:
        return (
            (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)
        ) - E / F

    RGB = f(RGB * exposure_bias, A, B, C, D, E, F)

    return RGB * (1 / as_float_array(f(linear_whitepoint, A, B, C, D, E, F)))
