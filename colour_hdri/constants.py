#!/usr/bin/env python
# -*- coding: utf-8 -*-

EXIF_TOOL = '/usr/local/bin/exiftool'

RAW_CONVERTER = '/usr/local/bin/dcraw'
RAW_CONVERSION_ARGUMENTS = '-t 0 -E -W -4 -T "{0}"'
RAW_D_CONVERSION_ARGUMENTS = '-t 0 -H 1 -r 1 1 1 1 -4 -q 3 -o 0 -T "{0}"'

DNG_CONVERTER = ('/Applications/Adobe DNG Converter.app/'
                 'Contents/MacOS/Adobe DNG Converter')
DNG_CONVERSION_ARGUMENTS = '-e -d "{0}" "{1}"'

DEFAUT_SOURCE_RAW_IMAGE_FORMATS = ('CR2', 'NEF', 'dng')
"""
:param DEFAUT_SOURCE_RAW_IMAGE_FORMATS: Default source raw image formats.
:type DEFAUT_SOURCE_RAW_IMAGE_FORMATS: tuple
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
