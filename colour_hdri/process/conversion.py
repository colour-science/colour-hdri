#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Conversion Process
==================

Defines the conversion process objects:

-   :func:`convert_raw_files_to_dng_files`
-   :func:`convert_dng_files_to_intermediate_files`
"""

from __future__ import division, unicode_literals

import logging
import os
import platform
import re
import shlex
import subprocess

from colour_hdri.utilities import path_exists

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RAW_CONVERTER',
           'RAW_CONVERSION_ARGUMENTS',
           'RAW_D_CONVERSION_ARGUMENTS',
           'DNG_CONVERTER',
           'DNG_CONVERSION_ARGUMENTS',
           'convert_raw_files_to_dng_files',
           'convert_dng_files_to_intermediate_files']

LOGGER = logging.getLogger(__name__)

RAW_CONVERTER = 'dcraw'
"""
Command line raw conversion application, usually Dave Coffin's *dcraw*.

RAW_CONVERTER : unicode
"""

RAW_CONVERSION_ARGUMENTS = '-t 0 -D -W -4 -T "{0}"'
"""
Arguments for the command line raw conversion application for non demosaiced
linear *tiff* file format output.

RAW_CONVERSION_ARGUMENTS : unicode
"""

RAW_D_CONVERSION_ARGUMENTS = '-t 0 -H 1 -r 1 1 1 1 -4 -q 3 -o 0 -T "{0}"'
"""
Arguments for the command line raw conversion application for demosaiced
linear *tiff* file format output.

RAW_D_CONVERSION_ARGUMENTS : unicode
"""

if platform.system() in ('Windows', 'Microsoft'):
    DNG_CONVERTER = 'C:\\Program Files (x86)\\Adobe\\Adobe DNG Converter.exe'
    """
    Command line *DNG* conversion application, usually *Adobe DNG Converter*.

    DNG_CONVERTER : unicode
    """
elif platform.system() == 'Darwin':
    DNG_CONVERTER = ('/Applications/Adobe DNG Converter.app/Contents/'
                     'MacOS/Adobe DNG Converter')
    """
    Command line *dng* conversion application, usually *Adobe DNG Converter*.

    DNG_CONVERTER : unicode
    """
else:
    DNG_CONVERTER = None
    """
    Command line *dng* conversion application, usually *Adobe DNG Converter*.

    DNG_CONVERTER : unicode
    """

DNG_CONVERSION_ARGUMENTS = '-e -d "{0}" "{1}"'
"""
Arguments for the command line *dng* conversion application.

DNG_CONVERSION_ARGUMENTS : unicode
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
                   if platform.system() in ("Windows", "Microsoft") else
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
                   if platform.system() in ("Windows", "Microsoft") else
                   True))

        subprocess.call(command)

        tiff_file = os.path.join(
            output_directory, os.path.basename(intermediate_file))
        if tiff_file != intermediate_file:
            path_exists(tiff_file) and os.remove(tiff_file)
            os.rename(intermediate_file, tiff_file)

        intermediate_files.append(tiff_file)

    return intermediate_files
