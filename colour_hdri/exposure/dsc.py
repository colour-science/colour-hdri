# -*- coding: utf-8 -*-
"""
Digital Still Camera Exposure
=============================

Defines various objects for modeling Digital Still Camera (DSC) exposure:

-   :func:`colour_hdri.focal_plane_exposure`
-   :func:`colour_hdri.arithmetic_mean_focal_plane_exposure`
-   :func:`colour_hdri.saturation_based_speed_focal_plane_exposure`
-   :func:`colour_hdri.exposure_index_values`
-   :func:`colour_hdri.exposure_value_100`
-   :func:`colour_hdri.photometric_exposure_scale_factor_Lagarde2014`

References
----------
-   :cite:`ISO2006` : ISO. (2006). INTERNATIONAL STANDARD ISO12232-2006 -
    Photography - Digital still cameras - Determination of exposure index, ISO
    speed ratings, standard output sensitivity, and recommended exposure index.
-   :cite:`Lagarde2014` : Lagarde, Sébastian, & de Rousiers, C. (2014). Moving
    Frostbite to Physically Based Rendering 3.0. Siggraph 2014, 119.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_float_array
from colour_hdri.exposure import (average_luminance,
                                  luminance_to_exposure_value)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'q_factor', 'focal_plane_exposure', 'arithmetic_mean_focal_plane_exposure',
    'saturation_based_speed_focal_plane_exposure', 'exposure_index_values',
    'exposure_value_100', 'photometric_exposure_scale_factor_Lagarde2014'
]


def q_factor(T=9 / 10, f_v=98 / 100, theta=10):
    """
    Computes the :math:`q` factor modeling the total lens vignetting and
    transmission attenuation.

    Parameters
    ----------
    T : array_like, optional
        Transmission factor of the lens :math:`T`.
    f_v : array_like, optional
        Vignetting factor :math:`f_v`.
    theta : array_like, optional
        Angle of image point off axis :math:`\\theta`.

    Returns
    -------
    ndarray
        :math:`q` factor.

    References
    ----------
    :cite:`ISO2006`

    Examples
    --------
    >>> q_factor()  # doctest: +ELLIPSIS
    0.6515748...
    """

    T = as_float_array(T)
    f_v = as_float_array(f_v)
    theta = as_float_array(theta)

    return np.pi / 4 * T * f_v * np.cos(np.radians(theta)) ** 4


def focal_plane_exposure(L, A, t, F, i, H_f, T=9 / 10, f_v=98 / 100, theta=10):
    """
    Computes the focal plane exposure :math:`H` in lux-seconds (:math:`lx.s`).

    Parameters
    ----------
    L : array_like
        Scene luminance :math:`L`, expressed in :math:`cd/m^2`.
    A : array_like
        Lens *F-Number* :math:`A`.
    t : array_like
        *Exposure Time* :math:`t`, expressed in seconds.
    F : array_like
        Lens focal length :math:`F`, expressed in meters.
    i : array_like
        Image distance :math:`i`, expressed in meters.
    H_f : array_like
        Focal plane flare exposure :math:`H_f`, expressed in lux-seconds
        (:math:`lx.s`).
    T : array_like, optional
        Transmission factor of the lens :math:`T`.
    f_v : array_like, optional
        Vignetting factor :math:`f_v`.
    theta : array_like, optional
        Angle of image point off axis :math:`\\theta`.

    Returns
    -------
    ndarray
        Focal plane exposure :math:`H` in lux-seconds (:math:`lx.s`).

    Notes
    -----
    -   Focal plane exposure is also named luminous exposure or photometric
        exposure and is time-integrated illuminance.
    -   Object distance :math:`o`, focal length :math:`F`, and image distance
        :math:`i` are related by the thin-lens equation:
        :math:`\\cfrac{1}{f}=\\cfrac{1}{o}+\\cfrac{1}{i}`
    -   This method ignores the *ISO* arithmetic speed :math:`S` and is not
        concerned with determining an appropriate minimum or maximum exposure
        level.

    References
    ----------
    :cite:`ISO2006`

    Examples
    --------
    >>> focal_plane_exposure(4000, 8, 1 / 250, 50 / 1000, 50 / 1000, 0.0015)
    ... # doctest: +ELLIPSIS
    0.1643937...
    """

    L = as_float_array(L)
    t = as_float_array(t)
    A = as_float_array(A)
    F = as_float_array(F)
    i = as_float_array(i)
    H_f = as_float_array(H_f)

    q = q_factor(T, f_v, theta)

    H = q * (L * t * F ** 2) / (A ** 2 * i ** 2) + H_f

    return H


def arithmetic_mean_focal_plane_exposure(L_a, A, t):
    """
    Computes the arithmetic mean focal plane exposure :math:`H_a` for a camera
    focused on infinity, :math:`H_f << H`, :math:`T=9/10`,
    :math:`\\theta =10^{\\circ}` and :math:`f_v=98/100`.

    Parameters
    ----------
    L_a : array_like
        Arithmetic scene luminance :math:`L_a`, expressed in :math:`cd/m^2`.
    A : array_like
        Lens *F-Number* :math:`A`.
    t : array_like
        *Exposure Time* :math:`t`, expressed in seconds.

    Returns
    -------
    ndarray
        Focal plane exposure :math:`H_a`.

    Notes
    -----
    -   Focal plane exposure is also named luminous exposure or photometric
        exposure and is time-integrated illuminance.
    -   Object distance :math:`o`, focal length :math:`F`, and image distance
        :math:`i` are related by the thin-lens equation:
        :math:`\\cfrac{1}{f}=\\cfrac{1}{o}+\\cfrac{1}{i}`
    -   This method ignores the *ISO* arithmetic speed :math:`S` and is not
        concerned with determining an appropriate minimum or maximum exposure
        level.

    References
    ----------
    :cite:`ISO2006`

    Examples
    --------
    >>> arithmetic_mean_focal_plane_exposure(4000, 8, 1 / 250)
    ... # doctest: +ELLIPSIS
    0.1628937...
    """

    H_a = focal_plane_exposure(L_a, A, t, 1, 1, 0, 9 / 10, 98 / 100, 10)

    return H_a


def saturation_based_speed_focal_plane_exposure(
        L,
        A,
        t,
        S,
        F=50 / 1000,
        i=1 / (-1 / 5 + 1 / (50 / 1000)),
        H_f=0,
        T=9 / 10,
        f_v=98 / 100,
        theta=10):
    """
    Computes the Saturation-Based Speed (SBS) focal plane exposure
    :math:`H_{SBS}` in lux-seconds (:math:`lx.s`).

    The model implemented by this definition is appropriate to simulate a
    physical camera in an offline or realtime renderer.

    Parameters
    ----------
    L : array_like
        Scene luminance :math:`L`, expressed in :math:`cd/m^2`.
    A : array_like
        Lens *F-Number* :math:`A`.
    t : array_like
        *Exposure Time* :math:`t`, expressed in seconds.
    S : array_like
        *ISO* arithmetic speed :math:`S`.
    F : array_like
        Lens focal length :math:`F`, expressed in meters.
    i : array_like
        Image distance :math:`i`, expressed in meters.
    H_f : array_like
        Focal plane flare exposure :math:`H_f`, expressed in lux-seconds
        (:math:`lx.s`).
    T : array_like, optional
        Transmission factor of the lens :math:`T`.
    f_v : array_like, optional
        Vignetting factor :math:`f_v`.
    theta : array_like, optional
        Angle of image point off axis :math:`\\theta`.

    Returns
    -------
    ndarray
        Saturation-Based Speed focal plane exposure :math:`H_{SBS}` in
        lux-seconds (:math:`lx.s`).

    Notes
    -----
    -   Focal plane exposure is also named luminous exposure or photometric
        exposure and is time-integrated illuminance.
    -   Object distance :math:`o`, focal length :math:`F`, and image distance
        :math:`i` are related by the thin-lens equation:
        :math:`\\cfrac{1}{f}=\\cfrac{1}{o}+\\cfrac{1}{i}`
    -   The image distance default value is that of an object located at 5m and
        imaged with a 50mm lens.
    -   The saturation based speed, :math:`S_{sat}`, of an electronic still
        picture camera is defined as: :math:`S_{sat}=\\cfrac{78}{H_{sat}}`
        where :math:`H_{sat}` is the minimum focal plane exposure, expressed in
        lux-seconds (:math:`lx.s`), that produces the maximum valid (not
        clipped or bloomed) camera output signal. This provides :math:`1/2`
        "stop" of headroom (41% additional headroom) for specular highlights
        above the signal level that would be obtained from a theoretical 100%
        reflectance object in the scene, so that a theoretical 141% reflectance
        object in the scene would produce a focal plane exposure of
        :math:`H_{sat}`.
    -   The focal plane exposure :math:`H_{SBS}` computed by this definition is
        almost equal to that given by scene luminance :math:`L` scaled with
        the output of :func:`colour_hdri.\
photometric_exposure_scale_factor_Lagarde2014` definition.

    References
    ----------
    :cite:`ISO2006`

    Examples
    --------
    >>> saturation_based_speed_focal_plane_exposure(  # doctest: +ELLIPSIS
    ...     4000, 8, 1 / 250, 400, 50 / 1000, 50 / 1000, 0.0015)
    0.8430446...
    """

    S = as_float_array(S)

    H = focal_plane_exposure(L, A, t, F, i, H_f, T, f_v, theta)

    H_SBS = H * S / 78

    return H_SBS


def exposure_index_values(H_a):
    """
    Computes the exposure index values :math:`I_{EI}` from given focal plane
    exposure :math:`H_a`.

    Parameters
    ----------
    H_a : array_like
        Focal plane exposure :math:`H_a`.

    Returns
    -------
    ndarray
        Exposure index values :math:`I_{EI}`.

    References
    ----------
    :cite:`ISO2006`

    Examples
    --------
    >>> exposure_index_values(0.1628937086212269)  # doctest: +ELLIPSIS
    61.3897251...
    """

    return 10 / H_a


def exposure_value_100(N, t, S):
    """
    Computes the exposure value :math:`EV100` from given relative aperture
    *F-Number* :math:`N`, *Exposure Time* :math:`t` and *ISO* arithmetic
    speed :math:`S`.

    Parameters
    ----------
    N : array_like
        Relative aperture *F-Number* :math:`N`.
    t : array_like
       *Exposure Time* :math:`t`.
    S : array_like
        *ISO* arithmetic speed :math:`S`.

    Returns
    -------
    ndarray
        Exposure value :math:`EV100`.

    References
    ----------
    :cite:`ISO2006`, :cite:`Lagarde2014`

    Notes
    -----
    -   The underlying implementation uses the
        :func:`colour_hdri.luminance_to_exposure_value` and
        :func:`colour_hdri.average_luminance` definitions with same fixed value
        for the *reflected light calibration constant* :math:`k` which cancels
        its scaling effect and produces a value equal to
        :math:`log_2(\\cfrac{N^2}{t}) - log_2(\\cfrac{S}{100})` as given in
        :cite:`Lagarde2014`.

    Examples
    --------
    >>> exposure_value_100(8, 1 / 250, 400)  # doctest: +ELLIPSIS
    11.9657842...
    """

    return luminance_to_exposure_value(average_luminance(N, t, S), 100)


def photometric_exposure_scale_factor_Lagarde2014(EV100,
                                                  T=9 / 10,
                                                  f_v=98 / 100,
                                                  theta=10):
    """
    Converts the exposure value :math:`EV100` to photometric exposure scale
    factor using *Lagarde and de Rousiers (2014)* formulation derived from the
    *ISO 12232:2006* *Saturation Based Sensitivity* (SBS) recommendation.

    The model implemented by this definition is appropriate to simulate a
    physical camera in an offline or realtime renderer.

    Parameters
    ----------
    T : array_like, optional
        Exposure value :math:`EV100`.
    T : array_like, optional
        Transmission factor of the lens :math:`T`.
    f_v : array_like, optional
        Vignetting factor :math:`f_v`.
    theta : array_like, optional
        Angle of image point off axis :math:`\\theta`.

    Returns
    -------
    ndarray
        Photometric exposure in lux-seconds (:math:`lx.s`).

    Notes
    -----
    -   The saturation based speed, :math:`S_{sat}`, of an electronic still
        picture camera is defined as: :math:`S_{sat}=\\cfrac{78}{H_{sat}}`
        where :math:`H_{sat}` is the minimum focal plane exposure, expressed in
        lux-seconds (:math:`lx.s`), that produces the maximum valid (not
        clipped or bloomed) camera output signal. This provides :math:`1/2`
        "stop" of headroom (41% additional headroom) for specular highlights
        above the signal level that would be obtained from a theoretical 100%
        reflectance object in the scene, so that a theoretical 141% reflectance
        object in the scene would produce a focal plane exposure of
        :math:`H_{sat}`.
    -   Scene luminance :math:`L` scaled with the photometric exposure value
        computed by this definition is almost equal to that given by the
        :func:`colour_hdri.saturation_based_speed_focal_plane_exposure`
        definition.

    References
    ----------
    :cite:`ISO2006`, :cite:`Lagarde2014`

    Examples
    --------
    >>> EV100 = exposure_value_100(8, 1 / 250, 400)
    >>> H = photometric_exposure_scale_factor_Lagarde2014(EV100)
    >>> print(H)  # doctest: +ELLIPSIS
    0.0002088...
    >>> H * 4000  # doctest: +ELLIPSIS
    0.8353523...
    """

    q = q_factor(T, f_v, theta)

    return 1 / (78 / (100 * q) * 2 ** EV100)
