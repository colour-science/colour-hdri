#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adobe DNG SDK Colour Processing
===============================

Defines various objects implementing *Adobe DNG SDK* colour processing:

-   :func:`xy_to_camera_neutral`
-   :func:`camera_neutral_to_xy`
-   :func:`XYZ_to_camera_space_matrix`
-   :func:`camera_space_to_XYZ_matrix`

The *Adobe DNG SDK* defines the following tags relevant for the current
implementation:

-   *CalibrationIlluminant1* : The illuminant used for the first set of
    colour calibration tags.
-   *CalibrationIlluminant2* : The illuminant used for an optional second set
    of colour calibration tags.
-   *ColorMatrix1* : *ColorMatrix1* defines a transformation matrix that
    converts XYZ values to reference camera native colour space values, under
    the first calibration illuminant.
-   *ColorMatrix2* : *ColorMatrix2* defines a transformation matrix that
    converts XYZ values to reference camera native colour space values, under
    the second calibration illuminant.
-   *CameraCalibration1* : *CameraCalibration1* defines a calibration matrix
    that transforms reference camera native space values to individual camera
    native space values under the first calibration illuminant.
    This matrix is stored separately from the matrix specified by the
    *ColorMatrix1* tag to allow raw converters to swap in replacement colour
    matrices based on *UniqueCameraModel* tag, while still taking advantage of
    any per-individual camera calibration performed by the camera manufacturer.
-   *CameraCalibration2* : *CameraCalibration2* defines a calibration matrix
    that transforms reference camera native space values to individual camera
    native space values under the second calibration illuminant.
    This matrix is stored separately from the matrix specified by the
    *ColorMatrix2* tag to allow raw converters to swap in replacement colour
    matrices based on *UniqueCameraModel* tag, while still taking advantage of
    any per-individual camera calibration performed by the camera manufacturer.
-   *ReductionMatrix1* : *ReductionMatrix1* defines a dimensionality reduction
    matrix for use as the first stage in converting colour camera native space
    values to XYZ values, under the first calibration illuminant. This tag may
    only be used if *ColorPlanes* is greater than 3.
-   *ReductionMatrix2* : *ReductionMatrix2* defines a dimensionality reduction
    matrix for use as the first stage in converting colour camera native space
    values to XYZ values, under the second calibration illuminant. This tag
    may only be used if *ColorPlanes* is greater than 3.
-   *AnalogBalance* : Normally the stored raw values are not white balanced,
    since any digital white balancing will reduce the dynamic range of the
    final image if the user decides to later adjust the white balance;
    however, if camera hardware is capable of white balancing the colour
    channels before the signal is digitized, it can improve the dynamic range
    of the final image.
    *AnalogBalance* defines the gain, either analog (recommended) or digital
    (not recommended) that has been applied the stored raw values.
-   *AsShotNeutral* : *AsShotNeutral* specifies the selected white balance at
    time of capture, encoded as the coordinates of a perfectly neutral colour
    in linear reference space values. The inclusion of this tag precludes the
    inclusion of the *AsShotWhiteXY* tag.
-   *AsShotWhiteXY* : *AsShotWhiteXY* specifies the selected white balance at
    time of capture, encoded as x-y chromaticity coordinates. The inclusion of
    this tag precludes the inclusion of the *AsShotNeutral* tag.
-   *ForwardMatrix1* : This tag defines a matrix that maps white balanced
    camera colours to XYZ D50 colours.
-   *ForwardMatrix2* : This tag defines a matrix that maps white balanced
    camera colours to XYZ D50 colours.

Notes
-----
-   At least one of the *ColorMatrix1* or *ColorMatrix2* tags must be included
    in the camera profile, the current implementation expects them to be passed
    as identity matrices if not included.
-   If the *ForwardMatrix1* or *ForwardMatrix2* tags are not included in the
    camera profile, the current implementation expects them to be passed as
    identity matrices.
-   The *ReductionMatrix1* and *ReductionMatrix2* tags are ignored by the
    current implementation which expects cameras with 3 colour planes.
-   *DNG 1.2.0.0* and later supports different companies creating the camera
    calibration tags using different reference cameras.
    When rendering a *DNG* file using a camera profile, it is important to
    know if the selected camera profile was designed using the same reference
    camera used to create the camera calibration tags. If so, then the camera
    calibration tags should be used. If not, then it is preferable to ignore
    the camera calibration tags and use identity matrices instead in order to
    minimize the worse case calibration mismatch error.
    This matching is done by comparing the *CameraCalibrationSignature* tag
    and the *ProfileCalibrationSignature* tag for the selected camera profile.
    If they match, then use the camera calibration tags. If not, then use
    identity matrices.
-   The Hue/Saturation/Value Mapping Table is ignored by the current
    implementation because deemed unsuitable. [3]_
-   The various matrices used in this module are extracted from a
    *Canon EOS 5D Mark II* camera.

References
----------
.. [1]  Adobe Systems. (2012). Digital Negative (DNG) Specification, 1–101.
.. [2]  Adobe Systems. (2015). Adobe DNG SDK 1.4. Retrieved from
        http://download.adobe.com/pub/adobe/dng/dng_sdk_1_4.zip
.. [3]  McGuffog, S. (2012). Hue Twists in DNG Camera Profiles. Retrieved
        October 29, 2016, from http://dcptool.sourceforge.net/Hue%20Twists.html
"""

from __future__ import division, unicode_literals

import numpy as np

from colour import (
    EPSILON,
    UCS_to_uv,
    XYZ_to_UCS,
    XYZ_to_xy,
    chromatic_adaptation_matrix_VonKries,
    dot_matrix,
    dot_vector,
    is_identity,
    linear_conversion,
    tstack,
    uv_to_CCT_Robertson1968,
    xy_to_XYZ)

from colour_hdri.models import ADOBE_DNG_XYZ_ILLUMINANT

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['interpolated_matrix',
           'xy_to_camera_neutral',
           'camera_neutral_to_xy',
           'XYZ_to_camera_space_matrix',
           'camera_space_to_XYZ_matrix']


def interpolated_matrix(CCT, CCT_1, CCT_2, M_1, M_2):
    """
    Computes the matrix interpolated from :math:`CCT_1` and :math:`CCT_2`
    correlated colour temperatures to respectively :math:`M_1` and :math:`M_2`
    colour matrices using given correlated colour temperature :math:`CCT`
    interpolation value.

    Parameters
    ----------
    CCT : numeric
        Correlated colour temperature :math:`CCT`.
    CCT_1 : numeric
        Correlated colour temperature :math:`CCT_1`.
    CCT_2 : numeric
        Correlated colour temperature :math:`CCT_2`.
    M_1 : array_like
        :math:`M_1` colour matrix.
    M_2 : array_like
        :math:`M_2` colour matrix.

    Returns
    -------
    ndarray
        Interpolated colour matrix :math:`M_i`.

    Notes
    -----
    -   The computation is performed in mired (MIcro REciprocal Degree,
        reciprocal megakelvin) :math:`MK^{-1}`.

    Examples
    --------
    >>> CCT = 5000
    >>> CCT_1 = 2850
    >>> CCT_2 = 6500
    >>> M_1 = np.array([
    ...     [0.5309, -0.0229, -0.0336],
    ...     [-0.6241, 1.3265, 0.3337],
    ...     [-0.0817, 0.1215, 0.6664]])
    >>> M_2 = np.array([
    ...     [0.4716, 0.0603, -0.0830],
    ...     [-0.7798, 1.5474, 0.2480],
    ...     [-0.1496, 0.1937, 0.6651]])
    >>> interpolated_matrix(CCT, CCT_1, CCT_2, M_1, M_2)  # doctest: +ELLIPSIS
    array([[ 0.4854908...,  0.0408106..., -0.0714282...],
           [-0.7433278...,  1.4956549...,  0.2680749...],
           [-0.1336946...,  0.1767874...,  0.6654045...]])
    """

    if CCT <= CCT_1:
        return M_1
    elif CCT >= CCT_2:
        return M_2
    else:
        return linear_conversion(
            1e6 / CCT, (1e6 / CCT_1, 1e6 / CCT_2), tstack((M_1, M_2)))


def xy_to_camera_neutral(xy,
                         CCT_calibration_illuminant_1,
                         CCT_calibration_illuminant_2,
                         M_color_matrix_1,
                         M_color_matrix_2,
                         M_camera_calibration_1,
                         M_camera_calibration_2,
                         analog_balance):
    """
    Converts given *xy* white balance chromaticity coordinates to
    *Camera Neutral* coordinates.

    Parameters
    ----------
    xy : array_like
        *xy* white balance chromaticity coordinates.
    CCT_calibration_illuminant_1 : numeric
        Correlated colour temperature of *CalibrationIlluminant1*.
    CCT_calibration_illuminant_2 : numeric
        Correlated colour temperature of *CalibrationIlluminant2*.
    M_color_matrix_1 : array_like
        *ColorMatrix1* tag matrix.
    M_color_matrix_2 : array_like
        *ColorMatrix2* tag matrix.
    M_camera_calibration_1 : array_like
        *CameraCalibration1* tag matrix.
    M_camera_calibration_2 : array_like
        *CameraCalibration2* tag matrix.
    analog_balance : array_like
        *AnalogBalance* tag vector.

    Returns
    -------
    ndarray
        *Camera Neutral* coordinates.

    References
    ----------
    .. [4]  Adobe Systems. (2012). Translating White Balance xy Coordinates to
            Camera Neutral Coordinates.
            In Digital Negative (DNG) Specification (p. 80).

    Examples
    --------
    >>> M_color_matrix_1 = np.array([
    ...     [0.5309, -0.0229, -0.0336],
    ...     [-0.6241, 1.3265, 0.3337],
    ...     [-0.0817, 0.1215, 0.6664]])
    >>> M_color_matrix_2 = np.array([
    ...     [0.4716, 0.0603, -0.0830],
    ...     [-0.7798, 1.5474, 0.2480],
    ...     [-0.1496, 0.1937, 0.6651]])
    >>> M_camera_calibration_1 = np.identity(3)
    >>> M_camera_calibration_2 = np.identity(3)
    >>> analog_balance = np.ones(3)
    >>> xy_to_camera_neutral(  # doctest: +ELLIPSIS
    ...     np.array([0.32816244, 0.34698169]),
    ...     2850,
    ...     6500,
    ...     M_color_matrix_1,
    ...     M_color_matrix_2,
    ...     M_camera_calibration_1,
    ...     M_camera_calibration_2,
    ...     analog_balance)
    array([ 0.4130699...,  1...        ,  0.646465...])
    """

    M_XYZ_to_camera = XYZ_to_camera_space_matrix(
        xy,
        CCT_calibration_illuminant_1,
        CCT_calibration_illuminant_2,
        M_color_matrix_1,
        M_color_matrix_2,
        M_camera_calibration_1,
        M_camera_calibration_2,
        analog_balance)

    camera_neutral = dot_vector(M_XYZ_to_camera, xy_to_XYZ(xy))
    camera_neutral /= camera_neutral[1]

    return camera_neutral


def camera_neutral_to_xy(camera_neutral,
                         CCT_calibration_illuminant_1,
                         CCT_calibration_illuminant_2,
                         M_color_matrix_1,
                         M_color_matrix_2,
                         M_camera_calibration_1,
                         M_camera_calibration_2,
                         analog_balance,
                         epsilon=EPSILON):
    """
    Converts given *Camera Neutral* coordinates to *xy* white balance
    chromaticity coordinates.

    Parameters
    ----------
    camera_neutral : array_like
        *Camera Neutral* coordinates.
    CCT_calibration_illuminant_1 : numeric
        Correlated colour temperature of *CalibrationIlluminant1*.
    CCT_calibration_illuminant_2 : numeric
        Correlated colour temperature of *CalibrationIlluminant2*.
    M_color_matrix_1 : array_like
        *ColorMatrix1* tag matrix.
    M_color_matrix_2 : array_like
        *ColorMatrix2* tag matrix.
    M_camera_calibration_1 : array_like
        *CameraCalibration1* tag matrix.
    M_camera_calibration_2 : array_like
        *CameraCalibration2* tag matrix.
    analog_balance : array_like
        *AnalogBalance* tag vector.
    epsilon : numeric, optional
        Threshold value for computation convergence.

    Returns
    -------
    ndarray
        *xy* white balance chromaticity coordinates.

    Raises
    ------
    RuntimeError
        If the given *Camera Neutral* coordinates did not converge to *xy*
        white balance chromaticity coordinates.

    References
    ----------
    .. [5]  Adobe Systems. (2012). Translating Camera Neutral Coordinates to
            White Balance xy Coordinates.
            In Digital Negative (DNG) Specification (pp. 80–81).

    Examples
    --------
    >>> M_color_matrix_1 = np.array([
    ...     [0.5309, -0.0229, -0.0336],
    ...     [-0.6241, 1.3265, 0.3337],
    ...     [-0.0817, 0.1215, 0.6664]])
    >>> M_color_matrix_2 = np.array([
    ...     [0.4716, 0.0603, -0.0830],
    ...     [-0.7798, 1.5474, 0.2480],
    ...     [-0.1496, 0.1937, 0.6651]])
    >>> M_camera_calibration_1 = np.identity(3)
    >>> M_camera_calibration_2 = np.identity(3)
    >>> analog_balance = np.ones(3)
    >>> camera_neutral_to_xy(  # doctest: +ELLIPSIS
    ...     np.array([0.413070, 1.000000, 0.646465]),
    ...     2850,
    ...     6500,
    ...     M_color_matrix_1,
    ...     M_color_matrix_2,
    ...     M_camera_calibration_1,
    ...     M_camera_calibration_2,
    ...     analog_balance)
    array([ 0.3281624...,  0.3469816...])
    """

    # Initial *xy* chromaticity coordinates guess.
    xy = np.array((1 / 3, 1 / 3))

    while True:
        xy_p = np.copy(xy)
        M_XYZ_to_camera = XYZ_to_camera_space_matrix(
            xy,
            CCT_calibration_illuminant_1,
            CCT_calibration_illuminant_2,
            M_color_matrix_1,
            M_color_matrix_2,
            M_camera_calibration_1,
            M_camera_calibration_2,
            analog_balance)

        XYZ = dot_vector(np.linalg.inv(M_XYZ_to_camera), camera_neutral)
        xy = XYZ_to_xy(XYZ)

        if np.abs(np.sum(xy_p - xy)) <= epsilon:
            return xy

    raise RuntimeError(
        '"Camera Neutral" coordinates "{0}" did not converge to "xy" white '
        'balance chromaticity coordinates!'.format(xy))


def XYZ_to_camera_space_matrix(xy,
                               CCT_calibration_illuminant_1,
                               CCT_calibration_illuminant_2,
                               M_color_matrix_1,
                               M_color_matrix_2,
                               M_camera_calibration_1,
                               M_camera_calibration_2,
                               analog_balance):
    """
    Returns the *CIE XYZ* to *Camera Space* matrix for given *xy* white balance
    chromaticity coordinates. [4]_

    Parameters
    ----------
    xy : array_like
        *xy* white balance chromaticity coordinates.
    CCT_calibration_illuminant_1 : numeric
        Correlated colour temperature of *CalibrationIlluminant1*.
    CCT_calibration_illuminant_2 : numeric
        Correlated colour temperature of *CalibrationIlluminant2*.
    M_color_matrix_1 : array_like
        *ColorMatrix1* tag matrix.
    M_color_matrix_2 : array_like
        *ColorMatrix2* tag matrix.
    M_camera_calibration_1 : array_like
        *CameraCalibration1* tag matrix.
    M_camera_calibration_2 : array_like
        *CameraCalibration2* tag matrix.
    analog_balance : array_like
        *AnalogBalance* tag vector.

    Returns
    -------
    ndarray
        *CIE XYZ* to *Camera Space* matrix.

    Notes
    -----
    -   The reference illuminant is D50 as defined per
        :attr:`ADOBE_DNG_XYZ_ILLUMINANT` attribute.

    Examples
    --------
    >>> M_color_matrix_1 = np.array([
    ...     [0.5309, -0.0229, -0.0336],
    ...     [-0.6241, 1.3265, 0.3337],
    ...     [-0.0817, 0.1215, 0.6664]])
    >>> M_color_matrix_2 = np.array([
    ...     [0.4716, 0.0603, -0.0830],
    ...     [-0.7798, 1.5474, 0.2480],
    ...     [-0.1496, 0.1937, 0.6651]])
    >>> M_camera_calibration_1 = np.identity(3)
    >>> M_camera_calibration_2 = np.identity(3)
    >>> analog_balance = np.ones(3)
    >>> XYZ_to_camera_space_matrix(  # doctest: +ELLIPSIS
    ...     np.array([0.34510414, 0.35162252]),
    ...     2850,
    ...     6500,
    ...     M_color_matrix_1,
    ...     M_color_matrix_2,
    ...     M_camera_calibration_1,
    ...     M_camera_calibration_2,
    ...     analog_balance)
    array([[ 0.4854908...,  0.0408106..., -0.0714282...],
           [-0.7433278...,  1.4956549...,  0.2680749...],
           [-0.1336946...,  0.1767874...,  0.6654045...]])
    """

    M_AB = np.diagflat(analog_balance)

    uv = UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(xy)))
    CCT, _D_uv = uv_to_CCT_Robertson1968(uv)

    if is_identity(M_color_matrix_1) or is_identity(M_color_matrix_2):
        M_CM = (M_color_matrix_1
                if is_identity(M_color_matrix_2) else
                M_color_matrix_2)
    else:
        M_CM = interpolated_matrix(
            CCT, CCT_calibration_illuminant_1, CCT_calibration_illuminant_2,
            M_color_matrix_1, M_color_matrix_2)

    M_CC = interpolated_matrix(
        CCT, CCT_calibration_illuminant_1, CCT_calibration_illuminant_2,
        M_camera_calibration_1, M_camera_calibration_2)

    M_XYZ_to_camera_space = dot_matrix(dot_matrix(M_AB, M_CC), M_CM)

    return M_XYZ_to_camera_space


def camera_space_to_XYZ_matrix(xy,
                               CCT_calibration_illuminant_1,
                               CCT_calibration_illuminant_2,
                               M_color_matrix_1,
                               M_color_matrix_2,
                               M_camera_calibration_1,
                               M_camera_calibration_2,
                               analog_balance,
                               M_forward_matrix_1,
                               M_forward_matrix_2,
                               chromatic_adaptation_transform='Bradford'):
    """
    Returns the *Camera Space* to *CIE XYZ* matrix for given *xy* white
    balance chromaticity coordinates.

    Parameters
    ----------
    xy : array_like
        *xy* white balance chromaticity coordinates.
    CCT_calibration_illuminant_1 : numeric
        Correlated colour temperature of *CalibrationIlluminant1*.
    CCT_calibration_illuminant_2 : numeric
        Correlated colour temperature of *CalibrationIlluminant2*.
    M_color_matrix_1 : array_like
        *ColorMatrix1* tag matrix.
    M_color_matrix_2 : array_like
        *ColorMatrix2* tag matrix.
    M_camera_calibration_1 : array_like
        *CameraCalibration1* tag matrix.
    M_camera_calibration_2 : array_like
        *CameraCalibration2* tag matrix.
    analog_balance : array_like
        *AnalogBalance* tag vector.
    M_forward_matrix_1 : array_like
        *ForwardMatrix1* tag matrix.
    M_forward_matrix_2 : array_like
        *ForwardMatrix2* tag matrix.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        Chromatic adaptation transform.

    Returns
    -------
    ndarray
        *Camera Space* to *CIE XYZ* matrix.

    Notes
    -----
    -   The reference illuminant is D50 as defined per
        :attr:`ADOBE_DNG_XYZ_ILLUMINANT` attribute.

    References
    ----------
    .. [6]  Adobe Systems. (2012). Camera to XYZ (D50) Transform.
            In Digital Negative (DNG) Specification (p. 81).

    Examples
    --------
    >>> M_color_matrix_1 = np.array([
    ...     [0.5309, -0.0229, -0.0336],
    ...     [-0.6241, 1.3265, 0.3337],
    ...     [-0.0817, 0.1215, 0.6664]])
    >>> M_color_matrix_2 = np.array([
    ...     [0.4716, 0.0603, -0.0830],
    ...     [-0.7798, 1.5474, 0.2480],
    ...     [-0.1496, 0.1937, 0.6651]])
    >>> M_camera_calibration_1 = np.identity(3)
    >>> M_camera_calibration_2 = np.identity(3)
    >>> analog_balance = np.ones(3)
    >>> M_forward_matrix_1 = np.array([
    ...     [0.8924, -0.1041, 0.1760],
    ...     [0.4351, 0.6621, -0.0972],
    ...     [0.0505, -0.1562, 0.9308]])
    >>> M_forward_matrix_2 = np.array([
    ...     [0.8924, -0.1041, 0.1760],
    ...     [0.4351, 0.6621, -0.0972],
    ...     [0.0505, -0.1562, 0.9308]])
    >>> camera_space_to_XYZ_matrix(  # doctest: +ELLIPSIS
    ...     np.array([0.32816244, 0.34698169]),
    ...     2850,
    ...     6500,
    ...     M_color_matrix_1,
    ...     M_color_matrix_2,
    ...     M_camera_calibration_1,
    ...     M_camera_calibration_2,
    ...     analog_balance,
    ...     M_forward_matrix_1,
    ...     M_forward_matrix_2)
    array([[ 2.1604087..., -0.1041...    ,  0.2722498...],
           [ 1.0533324...,  0.6621...    , -0.1503561...],
           [ 0.1222553..., -0.1562...    ,  1.4398304...]])
    """

    # *ForwardMatrix1* and *ForwardMatrix2* are not included in the camera
    # profile.
    if is_identity(M_forward_matrix_1) and is_identity(M_forward_matrix_2):
        M_camera_to_XYZ = np.linalg.inv(
            XYZ_to_camera_space_matrix(
                xy,
                CCT_calibration_illuminant_1,
                CCT_calibration_illuminant_2,
                M_color_matrix_1,
                M_color_matrix_2,
                M_camera_calibration_1,
                M_camera_calibration_2,
                analog_balance))
        M_CAT = chromatic_adaptation_matrix_VonKries(
            xy_to_XYZ(xy),
            xy_to_XYZ(ADOBE_DNG_XYZ_ILLUMINANT),
            chromatic_adaptation_transform)
        M_camera_space_to_XYZ = dot_matrix(M_CAT, M_camera_to_XYZ)
    else:
        uv = UCS_to_uv(XYZ_to_UCS(xy_to_XYZ(xy)))
        CCT, _D_uv = uv_to_CCT_Robertson1968(uv)

        M_CC = interpolated_matrix(
            CCT, CCT_calibration_illuminant_1, CCT_calibration_illuminant_2,
            M_camera_calibration_1, M_camera_calibration_2)

        # The reference implementation [2]_ diverges from the white-paper [1]_:
        # The reference implementation directly computes the camera neutral by
        # multiplying directly the interpolated colour matrix :math:`CM` with
        # the tristimulus values of the *xy* white balance chromaticity
        # coordinates.
        # The current implementation is based on the white-paper so that the
        # interpolated camera calibration matrix :math:`CC` and the
        # analog balance matrix :math:`AB` are accounted for.
        camera_neutral = xy_to_camera_neutral(
            xy,
            CCT_calibration_illuminant_1,
            CCT_calibration_illuminant_2,
            M_color_matrix_1,
            M_color_matrix_2,
            M_camera_calibration_1,
            M_camera_calibration_2,
            analog_balance)

        M_AB = np.diagflat(analog_balance)

        M_reference_neutral = dot_vector(
            np.linalg.inv(dot_matrix(M_AB, M_CC)),
            camera_neutral)
        M_D = np.linalg.inv(np.diagflat(M_reference_neutral))
        M_FM = interpolated_matrix(
            CCT, CCT_calibration_illuminant_1, CCT_calibration_illuminant_2,
            M_forward_matrix_1, M_forward_matrix_2)
        M_camera_space_to_XYZ = dot_matrix(
            dot_matrix(M_FM, M_D),
            np.linalg.inv(dot_matrix(M_AB, M_CC)))

    return M_camera_space_to_XYZ
