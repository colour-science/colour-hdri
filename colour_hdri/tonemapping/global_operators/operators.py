#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Global Tonemapping Operators
============================

Defines global tonemapping operators objects:

-   :func:`tonemapping_operator_simple`
-   :func:`tonemapping_operator_normalisation`
-   :func:`tonemapping_operator_gamma`
-   :func:`tonemapping_operator_logarithmic`
-   :func:`tonemapping_operator_exponential`
-   :func:`tonemapping_operator_logarithmic_mapping`
-   :func:`tonemapping_operator_exponentiation_mapping`
-   :func:`tonemapping_operator_Schlick1994`
-   :func:`tonemapping_operator_Tumblin1999`
-   :func:`tonemapping_operator_Reinhard2004`
-   :func:`tonemapping_operator_filmic`

See Also
--------
`Colour - HDRI - Examples: Global Tonemapping Operators Jupyter Notebook
<https://github.com/colour-science/colour-hdri/\
blob/master/colour_hdri/examples/examples_global_tonemapping_operators.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour import EPSILON, RGB_COLOURSPACES, RGB_luminance

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_average',
           'tonemapping_operator_simple',
           'tonemapping_operator_normalisation',
           'tonemapping_operator_gamma',
           'tonemapping_operator_logarithmic',
           'tonemapping_operator_exponential',
           'tonemapping_operator_logarithmic_mapping',
           'tonemapping_operator_exponentiation_mapping',
           'tonemapping_operator_Schlick1994',
           'tonemapping_operator_Tumblin1999',
           'tonemapping_operator_Reinhard2004',
           'tonemapping_operator_filmic']


def log_average(a, epsilon=EPSILON):
    """
    Computes the log average of given array.

    Parameters
    ----------
    a : array_like
        Array to compute the log average.
    epsilon : numeric, optional
        Constant to avoid singularities in computations.

    Returns
    -------
    numeric
        Array log average.

    Examples
    --------
    >>> log_average(np.linspace(0, 10, 10))  # doctest: +ELLIPSIS
    0.125071409675722
    """

    a = np.asarray(a)

    average = np.exp(np.average(np.log(a + epsilon)))

    return average


def tonemapping_operator_simple(RGB):
    """
    Performs given *RGB* array tonemapping using the simple method:
    :math:`\cfrac{RGB}{RGB + 1}`.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    References
    ----------
    .. [1]  Wikipedia. (n.d.). Tonemapping - Purpose and methods. Retrieved
            March 15, 2015, from
            http://en.wikipedia.org/wiki/Tone_mapping#Purpose_and_methods

    Examples
    --------
    >>> tonemapping_operator_simple(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]))  # doctest: +ELLIPSIS
    array([[[ 0.3245382...,  0.2601156...,  0.1911532...],
            [ 0.5830618...,  0.3567839...,  0.2808993...]],
    <BLANKLINE>
           [[ 0.8150290...,  0.6831692...,  0.5733340...],
            [ 0.8683127...,  0.7746486...,  0.6893211...]]])
    """

    RGB = np.asarray(RGB)

    return RGB / (RGB + 1)


def tonemapping_operator_normalisation(
        RGB,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs given *RGB* array tonemapping using the normalisation method.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    References
    ----------
    .. [2]  Banterle, F., Artusi, A., Debattista, K., & Chalmers, A. (2011).
            3.2.1 Simple Mapping Methods. In Advanced High Dynamic Range
            Imaging (pp. 38–41). A K Peters/CRC Press. ISBN:978-1568817194

    Examples
    --------
    >>> tonemapping_operator_normalisation(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]))  # doctest: +ELLIPSIS
    array([[[ 0.1194997...,  0.0874388...,  0.0587783...],
            [ 0.3478122...,  0.1379590...,  0.0971544...]],
    <BLANKLINE>
           [[ 1.0959009...,  0.5362936...,  0.3342115...],
            [ 1.6399638...,  0.8549608...,  0.5518382...]]])
    """

    RGB = np.asarray(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)

    RGB = RGB / L_max

    return RGB


def tonemapping_operator_gamma(RGB, gamma=1, EV=0):
    """
    Performs given *RGB* array tonemapping using the gamma and exposure
    correction method [2]_.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    gamma : numeric, optional
        :math:`\gamma` correction value.
    EV : numeric, optional
        Exposure adjustment value.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    Examples
    --------
    >>> tonemapping_operator_gamma(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]),
    ...      1.0, -3.0)  # doctest: +ELLIPSIS
    array([[[ 0.0600585...,  0.0439453...,  0.0295410...],
            [ 0.1748046...,  0.0693359...,  0.0488282...]],
    <BLANKLINE>
           [[ 0.5507817...,  0.2695323...,  0.1679692...],
            [ 0.8242187...,  0.4296892...,  0.2773447...]]])
    """

    RGB = np.asarray(RGB)

    exposure = 2 ** EV
    RGB = (exposure * RGB) ** (1 / gamma)

    return RGB


def tonemapping_operator_logarithmic(
        RGB,
        q=1,
        k=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs given *RGB* array tonemapping using the logarithmic method [2]_.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    q : numeric, optional
        :math:`q`.
    k : numeric, optional
        :math:`k`.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    Examples
    --------
    >>> tonemapping_operator_logarithmic(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]),
    ...       1.0, 25)  # doctest: +ELLIPSIS
    array([[[ 0.0884587...,  0.0647259...,  0.0435102...],
            [ 0.2278222...,  0.0903652...,  0.0636376...]],
    <BLANKLINE>
           [[ 0.4717487...,  0.2308565...,  0.1438669...],
            [ 0.5727396...,  0.2985858...,  0.1927235...]]])
    """

    RGB = np.asarray(RGB)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)
    L_d = np.log10(1 + L * q) / np.log10(1 + L_max * k)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_exponential(
        RGB,
        q=1,
        k=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs given *RGB* array tonemapping using the exponential method [2]_.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    q : numeric, optional
        :math:`q`.
    k : numeric, optional
        :math:`k`.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    Examples
    --------
    >>> tonemapping_operator_exponential(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]),
    ...       1.0, 25)  # doctest: +ELLIPSIS
    array([[[ 0.0148082...,  0.0108353...,  0.0072837...],
            [ 0.0428669...,  0.0170031...,  0.0119740...]],
    <BLANKLINE>
           [[ 0.1312736...,  0.0642404...,  0.0400338...],
            [ 0.1921684...,  0.1001830...,  0.0646635...]]])
    """

    RGB = np.asarray(RGB)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_a = log_average(L)
    L_d = 1 - np.exp(-(L * q) / (L_a * k))

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_logarithmic_mapping(
        RGB,
        p=1,
        q=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs given *RGB* array tonemapping using the logarithmic mapping
    method.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    p : numeric, optional
        :math:`p`.
    q : numeric, optional
        :math:`q`.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    References
    ----------
    .. [3]  Schlick, C. (1994). Quantization Techniques for Visualization of
            High Dynamic Range Pictures. Proceedings of the Fifth Eurographics
            Workshop on Rendering, (Section 5), 7–18.

    Examples
    --------
    >>> tonemapping_operator_logarithmic_mapping(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]))  # doctest: +ELLIPSIS
    array([[[ 0.2532899...,  0.1853341...,  0.1245857...],
            [ 0.6523387...,  0.2587489...,  0.1822179...]],
    <BLANKLINE>
           [[ 1.3507897...,  0.6610269...,  0.4119437...],
            [ 1.6399638...,  0.8549608...,  0.5518382...]]])
    """

    RGB = np.asarray(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)

    L_max = np.max(L)
    L_d = (np.log(1 + p * L) / np.log(1 + p * L_max)) ** (1 / q)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_exponentiation_mapping(
        RGB,
        p=1,
        q=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs given *RGB* array tonemapping using the exponentiation mapping
    method [3]_.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    p : numeric, optional
        :math:`p`.
    q : numeric, optional
        :math:`q`.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    Examples
    --------
    >>> tonemapping_operator_exponentiation_mapping(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]))  # doctest: +ELLIPSIS
    array([[[ 0.1194997...,  0.0874388...,  0.0587783...],
            [ 0.3478122...,  0.1379590...,  0.0971544...]],
    <BLANKLINE>
           [[ 1.0959009...,  0.5362936...,  0.3342115...],
            [ 1.6399638...,  0.8549608...,  0.5518382...]]])
    """

    RGB = np.asarray(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)
    L_d = (L / L_max) ** (p / q)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_Schlick1994(
        RGB,
        p=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs given *RGB* array tonemapping using *Schlick (1994)*
    method [2]_[3]_.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    p : numeric, optional
        :math:`p`.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    Examples
    --------
    >>> tonemapping_operator_Schlick1994(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]))  # doctest: +ELLIPSIS
    array([[[ 0.1194997...,  0.0874388...,  0.0587783...],
            [ 0.3478122...,  0.1379590...,  0.0971544...]],
    <BLANKLINE>
           [[ 1.0959009...,  0.5362936...,  0.3342115...],
            [ 1.6399638...,  0.8549608...,  0.5518382...]]])
    """

    # TODO: Implement automatic *p* and *non-uniform* computations support.

    RGB = np.asarray(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)
    L_d = (p * L) / (p * L - L + L_max)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_Tumblin1999(
        RGB,
        L_da=20,
        C_max=100,
        L_max=100,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs given *RGB* array tonemapping using
    *Tumblin, Hodgins and Guenter (1999)* method [2]_.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    L_da : numeric, optional
        :math:`L_{da}` display adaptation luminance, a mid-range display value.
    C_max : numeric, optional
        :math:`C_{max}` maximum contrast available from the display.
    L_max : numeric, optional
        :math:`L_{max}` maximum display luminance.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    References
    ----------
    .. [4]  Tumblin, J., Hodgins, J. K., & Guenter, B. K. (1999). Two methods
            for display of high contrast images. ACM Transactions on Graphics.
            doi:10.1145/300776.300783

    Examples
    --------
    >>> tonemapping_operator_Tumblin1999(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]))  # doctest: +ELLIPSIS
    array([[[ 0.0400492...,  0.0293043...,  0.0196990...],
            [ 0.1019768...,  0.0404489...,  0.0284852...]],
    <BLANKLINE>
           [[ 0.2490212...,  0.1218618...,  0.0759427...],
            [ 0.3408366...,  0.1776880...,  0.1146895...]]])
    """

    RGB = np.asarray(RGB)

    L_w = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)

    def f(x): return np.where(x > 100,
                              2.655,
                              1.855 + 0.4 * np.log10(x + 2.3 * 10 ** -5))

    L_wa = np.exp(np.mean(np.log(L_w + 2.3 * 10 ** -5)))
    g_d = f(L_da)
    g_w = f(L_wa)
    g_wd = g_w / (1.855 + 0.4 * np.log(L_da))

    mL_wa = np.sqrt(C_max) ** (g_wd - 1)

    L_d = mL_wa * L_da * (L_w / L_wa) ** (g_w / g_d)

    RGB = RGB * L_d[..., np.newaxis] / L_w[..., np.newaxis]
    RGB = RGB / L_max

    return RGB


def tonemapping_operator_Reinhard2004(
        RGB,
        f=0,
        m=0.3,
        a=0,
        c=0,
        colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Performs given *RGB* array tonemapping using *Reinhard and Devlin (2004)*
    method.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    f : numeric, optional
        :math:`f`.
    m : numeric, optional
        :math:`m`.
    a : numeric, optional
        :math:`a`.
    c : numeric, optional
        :math:`c`.
    colourspace : `colour.RGB_Colourspace`, optional
        *RGB* colourspace used for internal *Luminance* computation.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    References
    ----------
    .. [5]  Reinhard, E., & Devlin, K. (2005). Dynamic range reduction inspired
            by photoreceptor physiology. IEEE Transactions on Visualization and
            Computer Graphics, 11(1), 13–24. doi:10.1109/TVCG.2005.9

    Examples
    --------
    >>> tonemapping_operator_Reinhard2004(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]),
    ...     -10)  # doctest: +ELLIPSIS
    array([[[ 0.0216792...,  0.0159556...,  0.0107821...],
            [ 0.0605894...,  0.0249445...,  0.0176972...]],
    <BLANKLINE>
           [[ 0.1688972...,  0.0904532...,  0.0583584...],
            [ 0.2331935...,  0.1368456...,  0.0928316...]]])
    """

    RGB = np.asarray(RGB)

    C_av = np.array((np.average(RGB[..., 0]),
                     np.average(RGB[..., 1]),
                     np.average(RGB[..., 2])))

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)

    L_lav = log_average(L)
    L_min, L_max = np.min(L), np.max(L)

    f = np.exp(-f)

    m = (m if m > 0 else
         (0.3 + 0.7 * ((np.log(L_max) - L_lav) /
                       (np.log(L_max) - np.log(L_min)) ** 1.4)))

    I_l = (c * RGB + (1 - c)) * L[..., np.newaxis]
    I_g = c * C_av + (1 - c) * L_lav
    I_a = a * I_l + (1 - a) * I_g

    RGB = RGB / (RGB + (f * I_a) ** m)

    return RGB


def tonemapping_operator_filmic(RGB,
                                shoulder_strength=0.22,
                                linear_strength=0.3,
                                linear_angle=0.1,
                                toe_strength=0.2,
                                toe_numerator=0.01,
                                toe_denominator=0.3,
                                exposure_bias=2,
                                linear_whitepoint=11.2):
    """
    Performs given *RGB* array tonemapping using *Habble (2010)* method.

    Parameters
    ----------
    RGB : array_like
        *RGB* array to perform tonemapping onto.
    shoulder_strength : numeric, optional
        Shoulder strength.
    linear_strength : numeric, optional
        Linear strength.
    linear_angle : numeric, optional
        Linear angle.
    toe_strength : numeric, optional
        Toe strength.
    toe_numerator : numeric, optional
        Toe numerator.
    toe_denominator : numeric, optional
        Toe denominator.
    exposure_bias : numeric, optional
        Exposure bias.
    linear_whitepoint : numeric, optional
        Linear whitepoint.

    Returns
    -------
    ndarray
        Tonemapped *RGB* array.

    References
    ----------
    .. [6]  Habble, J. (2010). Filmic Tonemapping Operators. Retrieved March
            15, 2015, from http://filmicgames.com/archives/75
    .. [7]  Habble, J. (2010). Uncharted 2: HDR Lighting. Retrieved March 15,
            2015, from
            http://www.slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting

    Examples
    --------
    >>> tonemapping_operator_filmic(np.array(
    ...     [[[0.48046875, 0.35156256, 0.23632812],
    ...       [1.39843753, 0.55468757, 0.39062594]],
    ...      [[4.40625388, 2.15625895, 1.34375372],
    ...       [6.59375023, 3.43751395, 2.21875829]]]))  # doctest: +ELLIPSIS
    array([[[ 0.4507954...,  0.3619673...,  0.2617269...],
            [ 0.7567191...,  0.4933310...,  0.3911730...]],
    <BLANKLINE>
           [[ 0.9725554...,  0.8557374...,  0.7465713...],
            [ 1.0158782...,  0.9382937...,  0.8615161...]]])
    """

    RGB = np.asarray(RGB)

    A = shoulder_strength
    B = linear_strength
    C = linear_angle
    D = toe_strength
    E = toe_numerator
    F = toe_denominator

    def f(x, A, B, C, D, E, F): return (
        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F)

    RGB = f(RGB * exposure_bias, A, B, C, D, E, F)
    RGB = RGB * (1 / f(linear_whitepoint, A, B, C, D, E, F))

    return RGB
