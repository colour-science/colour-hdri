#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['EXIF_TOOL',
           'RAW_CONVERTER',
           'RAW_CONVERSION_ARGUMENTS',
           'RAW_D_CONVERSION_ARGUMENTS',
           'DNG_CONVERTER',
           'DNG_CONVERSION_ARGUMENTS',
           'DEFAULT_SOURCE_RAW_IMAGE_FORMATS',
           'DEFAULT_RAW_IMAGE_FORMAT',
           'DEFAULT_INTERMEDIATE_IMAGE_FORMAT',
           'DEFAULT_HDRI_IMAGE_FORMAT',
           'DEFAULT_LDRI_IMAGE_FORMAT']

EXIF_TOOL = 'exiftool'

RAW_CONVERTER = 'dcraw'
RAW_CONVERSION_ARGUMENTS = '-t 0 -D -W -4 -T "{0}"'
RAW_D_CONVERSION_ARGUMENTS = '-t 0 -H 1 -r 1 1 1 1 -4 -q 3 -o 0 -T "{0}"'

DNG_CONVERTER = ('/Applications/Adobe DNG Converter.app/'
                 'Contents/MacOS/Adobe DNG Converter')
DNG_CONVERSION_ARGUMENTS = '-e -d "{0}" "{1}"'

DEFAULT_SOURCE_RAW_IMAGE_FORMATS = ('CR2', 'NEF', 'dng')
"""
:param DEFAULT_SOURCE_RAW_IMAGE_FORMATS: Default source raw image formats.
:type DEFAULT_SOURCE_RAW_IMAGE_FORMATS: tuple
"""

DEFAULT_RAW_IMAGE_FORMAT = 'dng'
"""
:param DEFAULT_RAW_IMAGE_FORMAT: Default raw image format.
:type DEFAULT_RAW_IMAGE_FORMAT: unicode
"""
DEFAULT_INTERMEDIATE_IMAGE_FORMAT = 'tiff'
"""
:param DEFAULT_INTERMEDIATE_IMAGE_FORMAT: Default intermediate image format.
:type DEFAULT_INTERMEDIATE_IMAGE_FORMAT: unicode
"""
DEFAULT_HDRI_IMAGE_FORMAT = 'exr'
"""
:param DEFAULT_HDRI_IMAGE_FORMAT: Default hdr image format.
:type DEFAULT_HDRI_IMAGE_FORMAT: unicode
"""
DEFAULT_LDRI_IMAGE_FORMAT = 'jpg'
"""
:param DEFAULT_LDRI_IMAGE_FORMAT: Default ldr image format.
:type DEFAULT_LDRI_IMAGE_FORMAT: unicode
"""
