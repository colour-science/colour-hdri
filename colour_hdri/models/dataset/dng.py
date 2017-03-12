#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adobe DNG SDK Dataset
=====================

Defines various dataset objects for *Adobe DNG SDK*:

-   :attr:`ADOBE_DNG_XYZ_ILLUMINANT`
-   :attr:`DNG_ILLUMINANTS_CORRELATED_COLOUR_TEMPERATURE`
-   :attr:`LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS`
"""

from __future__ import division, unicode_literals

from colour import CaseInsensitiveMapping, ILLUMINANTS

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ADOBE_DNG_XYZ_ILLUMINANT',
           'DNG_ILLUMINANTS_CORRELATED_COLOUR_TEMPERATURE',
           'LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS']

ADOBE_DNG_XYZ_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'])
"""
*Adobe DNG SDK* default illuminant for *CIE XYZ* colourspace.

ADOBE_DNG_XYZ_ILLUMINANT : tuple
"""

DNG_ILLUMINANTS_CORRELATED_COLOUR_TEMPERATURE = (
    CaseInsensitiveMapping(
        {'Standard light A': 2850,
         'Tungsten (incandescent light)': 2850,
         'ISO studio tungsten': 3200,
         'D50': 5000,
         'D55': 5500,
         'Daylight': 5500,
         'Fine weather': 5500,
         'Flash': 5500,
         'Standard light B': 5500,
         'D65': 6500,
         'Standard light C': 6500,
         'Cloudy weather': 6500,
         'D75': 7500,
         'Shade': 7500,
         'Daylight fluorescent (D 5700 - 7100K)': (5700 + 7100) * 0.5,
         'Day white fluorescent (N 4600 - 5500K)': (4600 + 5400) * 0.5,
         'Cool white fluorescent (W 3800 - 4500K)': (3900 + 4500) * 0.5,
         'Fluorescent': (3900 + 4500) * 0.5,
         'White fluorescent (WW 3250 - 3800K)': (3200 + 3700) * 0.5,
         'Warm white fluorescent (L 2600 - 3250K)': (2600 + 3250) * 0.5}))
"""
*Adobe DNG SDK* illuminants correlated colour temperature.

References
----------
..[1]   Adobe Systems. (2015). Adobe DNG SDK 1.4 -
        dng_sdk_1_4/dng_sdk/source/dng_camera_profile.cpp -
        dng_camera_profile::IlluminantToTemperature.
        Retrieved from http://download.adobe.com/pub/adobe/dng/dng_sdk_1_4.zip

Notes
-----
-   The correlated colour temperature are given for the
    *CIE 1931 2 Degree Standard Observer*.

DNG_ILLUMINANTS_CORRELATED_COLOUR_TEMPERATURE : CaseInsensitiveMapping
"""

LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS = {
    1: 'Daylight',
    2: 'Fluorescent',
    3: 'Tungsten (incandescent light)',
    4: 'Flash',
    9: 'Fine weather',
    10: 'Cloudy weather',
    11: 'Shade',
    12: 'Daylight fluorescent (D 5700 - 7100K)',
    13: 'Day white fluorescent (N 4600 - 5500K)',
    14: 'Cool white fluorescent (W 3800 - 4500K)',
    15: 'White fluorescent (WW 3250 - 3800K)',
    16: 'Warm white fluorescent (L 2600 - 3250K)',
    17: 'Standard light A',
    18: 'Standard light B',
    19: 'Standard light C',
    20: 'D55',
    21: 'D65',
    22: 'D75',
    23: 'D50',
    24: 'ISO studio tungsten',
    255: 'Other'}
"""
*Adobe DNG SDK* *LightSource Tag* indexes mapping to illuminants.

References
----------
..[2]   Adobe Systems. (2015). Adobe DNG SDK 1.4 -
        dng_sdk_1_4/dng_sdk/source/dng_tag_values.h - LightSource tag.
        Retrieved from http://download.adobe.com/pub/adobe/dng/dng_sdk_1_4.zip

LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS : CaseInsensitiveMapping
"""
