#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adobe DNG SDK Conversion Process
================================

Defines various objects implementing raw conversion based on *Adobe DNG SDK*
and *dcraw*:

-   :func:`convert_raw_files_to_dng_files`
-   :func:`convert_dng_files_to_intermediate_files`
-   :func:`read_dng_files_exif_tags`
"""

from __future__ import division, unicode_literals

import logging
import numpy as np
import os
import platform
import re
import shlex
import subprocess
from copy import deepcopy

from colour import CaseInsensitiveMapping, warning

from colour_hdri.utilities import (
    ExifTag,
    parse_exif_array,
    parse_exif_numeric,
    parse_exif_string,
    path_exists,
    read_exif_tags)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RAW_CONVERTER',
           'RAW_CONVERSION_ARGUMENTS',
           'RAW_D_CONVERSION_ARGUMENTS',
           'DNG_CONVERTER',
           'DNG_CONVERSION_ARGUMENTS',
           'DNG_EXIF_TAGS_BINDING',
           'convert_raw_files_to_dng_files',
           'convert_dng_files_to_intermediate_files',
           'read_dng_files_exif_tags']

LOGGER = logging.getLogger(__name__)

RAW_CONVERTER = 'dcraw'
"""
Command line raw conversion application, usually Dave Coffin's *dcraw*.

RAW_CONVERTER : unicode
"""

RAW_CONVERSION_ARGUMENTS = '-t 0 -D -W -4 -T "{0}"'
if platform.system() in ('Windows', 'Microsoft'):
    RAW_CONVERSION_ARGUMENTS = RAW_CONVERSION_ARGUMENTS.replace('"', '')
"""
Arguments for the command line raw conversion application for non demosaiced
linear *tiff* file format output.

RAW_CONVERSION_ARGUMENTS : unicode
"""

RAW_D_CONVERSION_ARGUMENTS = '-t 0 -H 1 -r 1 1 1 1 -4 -q 3 -o 0 -T "{0}"'
if platform.system() in ('Windows', 'Microsoft'):
    RAW_D_CONVERSION_ARGUMENTS = RAW_D_CONVERSION_ARGUMENTS.replace('"', '')
"""
Arguments for the command line raw conversion application for demosaiced
linear *tiff* file format output.

RAW_D_CONVERSION_ARGUMENTS : unicode
"""

if platform.system() == 'Darwin':
    DNG_CONVERTER = ('/Applications/Adobe DNG Converter.app/Contents/'
                     'MacOS/Adobe DNG Converter')
elif platform.system() in ('Windows', 'Microsoft'):
    DNG_CONVERTER = 'C:\\Program Files (x86)\\Adobe\\Adobe DNG Converter.exe'
else:
    DNG_CONVERTER = None
    warning('"Adobe DNG Converter" is not available on your platform!')
"""
Command line *DNG* conversion application, usually *Adobe DNG Converter*.

DNG_CONVERTER : unicode
"""

DNG_CONVERSION_ARGUMENTS = '-e -d "{0}" "{1}"'
if platform.system() in ('Windows', 'Microsoft'):
    DNG_CONVERSION_ARGUMENTS = DNG_CONVERSION_ARGUMENTS.replace('"', '')
"""
Arguments for the command line *dng* conversion application.

DNG_CONVERSION_ARGUMENTS : unicode
"""

DNG_EXIF_TAGS_BINDING = CaseInsensitiveMapping({
    'EXIF': CaseInsensitiveMapping({
        'Make': (parse_exif_string, None),
        'Camera Model Name': (parse_exif_string, None),
        'Camera Serial Number': (parse_exif_string, None),
        'Lens Model': (parse_exif_string, None),
        'DNG Lens Info': (parse_exif_string, None),
        'Focal Length': (parse_exif_numeric, None),
        'Exposure Time': (parse_exif_numeric, None),
        'F Number': (parse_exif_numeric, None),
        'ISO': (parse_exif_numeric, None),
        'CFA Pattern 2': (lambda x: parse_exif_array(x, np.int_), None),
        'CFA Plane Color': (lambda x: parse_exif_array(x, np.int_), None),
        'Black Level Repeat Dim': (
            lambda x: parse_exif_array(x, np.int_), None),
        'Black Level': (
            lambda x: parse_exif_array(x, np.int_), None),
        'White Level': (
            lambda x: parse_exif_numeric(x, np.int_), None),
        'Samples Per Pixel': (lambda x: parse_exif_numeric(x, np.int_), None),
        'Active Area': (lambda x: parse_exif_array(x, np.int_), None),
        'Orientation': (lambda x: parse_exif_numeric(x, np.int_), None),
        'Camera Calibration Sig': (parse_exif_string, None),
        'Profile Calibration Sig': (parse_exif_string, None),
        'Calibration Illuminant 1': (
            lambda x: parse_exif_numeric(x, np.int_), 17),
        'Calibration Illuminant 2': (
            lambda x: parse_exif_numeric(x, np.int_), 21),
        'Color Matrix 1': (
            lambda x: parse_exif_array(x, np.float_, (3, 3)),
            '1 0 0 0 1 0 0 0 1'),
        'Color Matrix 2': (
            lambda x: parse_exif_array(x, np.float_, (3, 3)),
            '1 0 0 0 1 0 0 0 1'),
        'Camera Calibration 1': (
            lambda x: parse_exif_array(x, np.float_, (3, 3)),
            '1 0 0 0 1 0 0 0 1'),
        'Camera Calibration 2': (
            lambda x: parse_exif_array(x, np.float_, (3, 3)),
            '1 0 0 0 1 0 0 0 1'),
        'Analog Balance': (
            lambda x: parse_exif_array(x, np.float_), '1 1 1'),
        'Reduction Matrix 1': (
            lambda x: parse_exif_array(x, np.float_, (3, 3)),
            '1 0 0 0 1 0 0 0 1'),
        'Reduction Matrix 2': (
            lambda x: parse_exif_array(x, np.float_, (3, 3)),
            '1 0 0 0 1 0 0 0 1'),
        'Forward Matrix 1': (
            lambda x: parse_exif_array(x, np.float_, (3, 3)),
            '1 0 0 0 1 0 0 0 1'),
        'Forward Matrix 2': (
            lambda x: parse_exif_array(x, np.float_, (3, 3)),
            '1 0 0 0 1 0 0 0 1'),
        'As Shot Neutral': (
            lambda x: parse_exif_array(x, np.float_), '1 1 1'),
        'Baseline Exposure': (
            lambda x: parse_exif_numeric(x, np.float_), None),
        'Baseline Noise': (
            lambda x: parse_exif_numeric(x, np.float_), None)})})
"""
Exif tags binding for a *dng* file.

DNG_EXIF_TAGS_BINDING : CaseInsensitiveMapping
"""


def convert_raw_files_to_dng_files(raw_files, output_directory):
    """
    Converts given raw files to *dng* files using given output directory.

    Parameters
    ----------
    raw_files : array_like
        Raw files to convert to *dng* files.
    output_directory : unicode
        Output directory.

    Returns
    -------
    list
        *dng* files.
    """

    dng_files = []
    for raw_file in raw_files:
        raw_file_extension = os.path.splitext(raw_file)[1]
        dng_file = os.path.join(output_directory, os.path.basename(
            re.sub('{0}$'.format(raw_file_extension), '.dng', raw_file)))

        path_exists(dng_file) and os.remove(dng_file)

        LOGGER.info(
            'Converting "{0}" file to "{1}" file.'.format(raw_file, dng_file))

        command = [DNG_CONVERTER] + shlex.split(
            DNG_CONVERSION_ARGUMENTS.format(output_directory, raw_file),
            posix=(False
                   if platform.system() in ('Windows', 'Microsoft') else
                   True))

        subprocess.call(command)

        dng_files.append(dng_file)

    return dng_files


def convert_dng_files_to_intermediate_files(dng_files,
                                            output_directory,
                                            demosaicing=False):
    """
    Converts given *dng* files to intermediate *tiff* files using given output
    directory.

    Parameters
    ----------
    dng_files : array_like
        *dng* files to convert to intermediate *tiff* files.
    output_directory : str
        Output directory.
    demosaicing : bool
        Perform demosaicing on conversion.

    Returns
    -------
    list
        Intermediate *tiff* files.
    """

    intermediate_files = []
    for dng_file in dng_files:
        intermediate_file = re.sub('\.dng$', '.tiff', dng_file)

        path_exists(intermediate_file) and os.remove(intermediate_file)

        LOGGER.info('Converting "{0}" file to "{1}" file.'.format(
            dng_file, intermediate_file))

        raw_conversion_arguments = (RAW_D_CONVERSION_ARGUMENTS
                                    if demosaicing else
                                    RAW_CONVERSION_ARGUMENTS)
        command = [RAW_CONVERTER] + shlex.split(
            raw_conversion_arguments.format(dng_file),
            posix=(False
                   if platform.system() in ('Windows', 'Microsoft') else
                   True))

        subprocess.call(command)

        tiff_file = os.path.join(
            output_directory, os.path.basename(intermediate_file))
        if tiff_file != intermediate_file:
            path_exists(tiff_file) and os.remove(tiff_file)
            os.rename(intermediate_file, tiff_file)

        intermediate_files.append(tiff_file)

    return intermediate_files


def read_dng_files_exif_tags(dng_files,
                             exif_tags_binding=DNG_EXIF_TAGS_BINDING):
    """
    Reads given *dng* files exif tags using given binding.

    Parameters
    ----------
    dng_files : array_like
        *dng* files to read the exif tags from.
    exif_tags_binding : dict_like
        Exif tags binding.

    Returns
    -------
    list
        *dng* files exif tags.
    """

    dng_files_exif_tags = []
    for dng_file in dng_files:
        exif_tags = read_exif_tags(dng_file)
        binding = deepcopy(exif_tags_binding)
        for group, tags in binding.items():
            for tag in tags:
                exif_tag = exif_tags[group].get(tag)
                parser = binding[group][tag][0]
                if exif_tag is None:
                    default = binding[group][tag][1]
                    binding[group][tag] = (
                        default if default is None else
                        parser(ExifTag(value=binding[group][tag][1])))
                else:
                    binding[group][tag] = parser(exif_tag[0])

        dng_files_exif_tags.append(binding)

    return dng_files_exif_tags
