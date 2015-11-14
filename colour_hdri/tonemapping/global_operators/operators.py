#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Global Tonemapping Operators
============================

"""

from __future__ import division, unicode_literals
import numpy as np
from colour import EPSILON, RGB_COLOURSPACES, RGB_luminance

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = []


def conform(a):
    return np.clip(np.nan_to_num(a), 0, 1)


def log_average(a, epsilon=EPSILON):
    a = np.asarray(a)

    average = np.exp(np.average(np.log(a + epsilon)))

    return average


def tonemapping_operator_simple(RGB):
    # Wikipedia. (n.d.). Tonemapping - Purpose and methods. Retrieved March 15, 2015, from http://en.wikipedia.org/wiki/Tone_mapping#Purpose_and_methods

    RGB = np.asarray(RGB)

    return RGB / (RGB + 1)


def tonemapping_operator_normalisation(
        RGB,
        colourspace=RGB_COLOURSPACES['sRGB']):
    # Banterle, F., Artusi, A., Debattista, K., & Chalmers, A. (2011). 3.2.1 Simple Mapping Methods. In Advanced High Dynamic Range Imaging (pp. 38–41). A K Peters/CRC Press. ISBN:978-1568817194
    RGB = np.asarray(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)
    RGB = RGB / L_max

    RGB = conform(RGB)

    return RGB


def tonemapping_operator_gamma(RGB, gamma=1, f_stop=0):
    # Banterle, F., Artusi, A., Debattista, K., & Chalmers, A. (2011). 3.2.1 Simple Mapping Methods. In Advanced High Dynamic Range Imaging (pp. 38–41). A K Peters/CRC Press. ISBN:978-1568817194
    RGB = np.asarray(RGB)

    exposure = 2 ** f_stop
    RGB = (exposure * RGB) ** (1 / gamma)

    RGB = conform(RGB)

    return RGB


def tonemapping_operator_logarithmic(
        RGB,
        q=1,
        k=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    # Banterle, F., Artusi, A., Debattista, K., & Chalmers, A. (2011). 3.2.1 Simple Mapping Methods. In Advanced High Dynamic Range Imaging (pp. 38–41). A K Peters/CRC Press. ISBN:978-1568817194
    RGB = np.asarray(RGB)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)
    L_d = np.log10(1 + L * q) / np.log10(1 + L_max * k)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    RGB = conform(RGB)

    return RGB


def tonemapping_operator_exponential(
        RGB,
        q=1,
        k=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    # Banterle, F., Artusi, A., Debattista, K., & Chalmers, A. (2011). 3.2.1 Simple Mapping Methods. In Advanced High Dynamic Range Imaging (pp. 38–41). A K Peters/CRC Press. ISBN:978-1568817194
    RGB = np.asarray(RGB)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_a = log_average(L)
    L_d = 1 - np.exp(-(L * q) / (L_a * k));

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    RGB = conform(RGB)

    return RGB


def tonemapping_operator_logarithmic_mapping(
        RGB,
        p=1,
        q=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    # Schlick, C. (1994). Quantization Techniques for Visualization of High Dynamic Range Pictures. Proceedings of the Fifth Eurographics Workshop on Rendering, (Section 5), 7–18.
    RGB = np.asarray(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)

    L_max = np.max(L)
    L_d = (np.log(1 + p * L) / np.log(1 + p * L_max)) ** (1 / q)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    RGB = conform(RGB)

    return RGB


def tonemapping_operator_exponentiation_mapping(
        RGB,
        p=1,
        q=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    # Schlick, C. (1994). Quantization Techniques for Visualization of High Dynamic Range Pictures. Proceedings of the Fifth Eurographics Workshop on Rendering, (Section 5), 7–18.
    RGB = np.asarray(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)
    L_d = (L / L_max) ** (p / q)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    RGB = conform(RGB)

    return RGB


def tonemapping_operator_Schlick94(
        RGB,
        p=1,
        colourspace=RGB_COLOURSPACES['sRGB']):
    # Schlick, C. (1994). Quantization Techniques for Visualization of High Dynamic Range Pictures. Proceedings of the Fifth Eurographics Workshop on Rendering, (Section 5), 7–18.
    # Banterle, F., Artusi, A., Debattista, K., & Chalmers, A. (2011). 3.2.3 Quantization Techniques. Advanced High Dynamic Range Imaging. A K Peters/CRC Press. ISBN:978-1568817194
    # Implement automatic *p* and *non-uniform* computations support.

    RGB = np.asarray(RGB)

    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)
    L_d = (p * L) / (p * L - L + L_max)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    RGB = conform(RGB)

    return RGB


def contrast_sensitivity_function_Tumblin98(L_a):
    # Tumblin, J., Hodgins, J. K., & Guenter, B. K. (1999). Two methods for display of high contrast images. ACM Transactions on Graphics. doi:10.1145/300776.300783
    # Banterle, F., Artusi, A., Debattista, K., & Chalmers, A. (2011). 3.2.2 Brightness Reproduction. Advanced High Dynamic Range Imaging. A K Peters/CRC Press. ISBN:978-1568817194
    L_a = np.asarray(L_a)

    gamma = np.where(L_a > 100,
                     2.655,
                     1.855 + 0.4 * np.log10(L_a + 2.3 * 10 ** -5))

    return gamma


def tonemapping_operator_Tumblin98(
        RGB,
        L_da=20,
        C_max=100,
        L_max=100,
        colourspace=RGB_COLOURSPACES['sRGB']):
    RGB = np.asarray(RGB)

    L_w = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)

    L_wa = np.exp(np.mean(np.log(L_w + 2.3 * 10 ** -5)))
    g_d = contrast_sensitivity_function_Tumblin98(L_da)
    g_w = contrast_sensitivity_function_Tumblin98(L_wa)
    g_wd = g_w / (1.855 + 0.4 * np.log(L_da))

    mL_wa = np.sqrt(C_max) ** (g_wd - 1)

    L_d = mL_wa * L_da * (L_w / L_wa) ** (g_w / g_d)

    RGB = RGB * L_d[..., np.newaxis] / L_w[..., np.newaxis]
    RGB = RGB / L_max

    RGB = conform(RGB)

    return RGB


def tonemapping_operator_Reinhard04(
        RGB,
        f=0,
        m=0.3,
        a=0,
        c=0,
        colourspace=RGB_COLOURSPACES['sRGB']):
    # Reinhard, E., & Devlin, K. (2005). Dynamic range reduction inspired by photoreceptor physiology. IEEE Transactions on Visualization and Computer Graphics, 11(1), 13–24. doi:10.1109/TVCG.2005.9
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

    RGB = conform(RGB)

    return RGB


def tonemapping_operator_filmic_function(x, A, B, C, D, E, F):
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def tonemapping_operator_filmic(RGB,
                                shoulder_strength=0.22,
                                linear_strength=0.3,
                                linear_angle=0.1,
                                toe_strength=0.2,
                                toe_numerator=0.01,
                                toe_denominator=0.3,
                                exposure_bias=2,
                                linear_whitepoint=11.2):
    # Habble, J. (2010). Filmic Tonemapping Operators. Retrieved March 15, 2015, from http://filmicgames.com/archives/75
    # Habble, J. (2010). Uncharted 2: HDR Lighting. Retrieved March 15, 2015, from http://www.slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting
    RGB = np.asarray(RGB)

    A = shoulder_strength
    B = linear_strength
    C = linear_angle
    D = toe_strength
    E = toe_numerator
    F = toe_denominator

    RGB = tonemapping_operator_filmic_function(
        RGB * exposure_bias, A, B, C, D, E, F)
    RGB = RGB * (1 / tonemapping_operator_filmic_function(
        linear_whitepoint, A, B, C, D, E, F))

    RGB = conform(RGB)

    return RGB
