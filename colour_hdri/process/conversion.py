#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import logging
import os
import platform
import re
import shlex
import subprocess

from colour_hdri.utilities import path_exists

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LOGGER',
           'RAW_CONVERTER',
           'RAW_CONVERSION_ARGUMENTS',
           'RAW_D_CONVERSION_ARGUMENTS',
           'DNG_CONVERTER',
           'DNG_CONVERSION_ARGUMENTS',
           'DEFAULT_SOURCE_RAW_IMAGE_FORMATS',
           'DEFAULT_RAW_IMAGE_FORMAT',
           'DEFAULT_INTERMEDIATE_IMAGE_FORMAT'
           'convert_raw_files_to_dng_files',
           'convert_dng_files_to_intermediate_files']

LOGGER = logging.getLogger(__name__)

RAW_CONVERTER = 'dcraw'
RAW_CONVERSION_ARGUMENTS = '-t 0 -D -W -4 -T "{0}"'
RAW_D_CONVERSION_ARGUMENTS = '-t 0 -H 1 -r 1 1 1 1 -4 -q 3 -o 0 -T "{0}"'

if platform.system() in ('Windows', 'Microsoft'):
    DNG_CONVERTER = 'C:\\Program Files (x86)\\Adobe\\Adobe DNG Converter.exe'
elif platform.system() == 'Darwin':
    DNG_CONVERTER = ('/Applications/Adobe DNG Converter.app/Contents/'
                     'MacOS/Adobe DNG Converter')

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


def convert_raw_files_to_dng_files(raw_files, output_directory):
    """
    Converts given raw files to dng files using given output directory.
    :param raw_files: Raw files to convert.
    :type raw_files: list
    :param output_directory: Output directory.
    :type output_directory: unicode
    :return: Intermediate files.
    :rtype: list
    """

    dng_files = []
    for raw_file in raw_files:
        dng_file = os.path.join(output_directory, os.path.basename(
            re.sub('{0}$'.format(os.path.splitext(raw_file)[1]),
                   '.{0}'.format(DEFAULT_RAW_IMAGE_FORMAT),
                   raw_file)))

        path_exists(dng_file) and os.remove(dng_file)

        LOGGER.info(
            'Converting "{0}" file to "{1}" file.'.format(raw_file, dng_file))

        command = [DNG_CONVERTER] + shlex.split(
            DNG_CONVERSION_ARGUMENTS.format(output_directory, raw_file),
            posix=(False
                   if platform.system() in ("Windows", "Microsoft") else
                   True))

        subprocess.call(command)

        dng_files.append(dng_file)

    return dng_files


def convert_dng_files_to_intermediate_files(dng_files,
                                            output_directory,
                                            demosaicing=False):
    """
    Converts given dng files to intermediate files using given output
    directory.
    :param dng_files: Dng files to convert.
    :type dng_files: list
    :param output_directory: Output directory.
    :type output_directory: str
    :param demosaicing: Perform demosaicing.
    :type demosaicing: bool
    :return: Intermediate files.
    :rtype: list
    """

    intermediate_files = []
    for dng_file in dng_files:
        interim_tiff_file = re.sub(
            '\.{0}$'.format(DEFAULT_RAW_IMAGE_FORMAT),
            '.{0}'.format(DEFAULT_INTERMEDIATE_IMAGE_FORMAT),
            dng_file)

        path_exists(interim_tiff_file) and os.remove(interim_tiff_file)

        LOGGER.info('Converting "{0}" file to "{1}" file.'.format(
            dng_file, interim_tiff_file))

        raw_conversion_arguments = (RAW_D_CONVERSION_ARGUMENTS
                                    if demosaicing else
                                    RAW_CONVERSION_ARGUMENTS)
        command = [RAW_CONVERTER] + shlex.split(
            raw_conversion_arguments.format(dng_file),
            posix=(False
                   if platform.system() in ("Windows", "Microsoft") else
                   True))

        subprocess.call(command)

        tiff_file = os.path.join(output_directory,
                                 os.path.basename(interim_tiff_file))
        if tiff_file != interim_tiff_file:
            path_exists(tiff_file) and os.remove(tiff_file)
            os.rename(interim_tiff_file, tiff_file)

        intermediate_files.append(tiff_file)

    return intermediate_files
