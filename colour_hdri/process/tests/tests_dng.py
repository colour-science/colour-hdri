# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.process.adobe_dng` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import platform
import shutil
import tempfile
import unittest

from colour import read_image

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.process import (
    convert_raw_files_to_dng_files,
    convert_dng_files_to_intermediate_files,
    read_dng_files_exif_tags)
from colour_hdri.utilities import filter_files

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FROBISHER_001_DIRECTORY',
           'PROCESS_DIRECTORY',
           'RAW_IMAGES',
           'TestConvertRawFilesToDngFiles',
           'TestConvertDngFilesToIntermediateFiles']

FROBISHER_001_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'frobisher_001')

PROCESS_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'colour_hdri', 'process')

RAW_IMAGES = filter_files(FROBISHER_001_DIRECTORY, ('CR2',))


class TestConvertRawFilesToDngFiles(unittest.TestCase):
    """
    Defines :func:`colour_hdri.process.adobe_dng.\
convert_raw_files_to_dng_files` definition unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_convert_raw_files_to_dng_files(self):
        """
        Tests :func:`colour_hdri.process.adobe_dng.\
convert_raw_files_to_dng_files` definition.
        """

        if platform.system() not in ('Windows', 'Microsoft', 'Darwin'):
            # *Adobe DNG Converter* is not available on Linux, thus we skip
            # that unit test.
            return

        reference_dng_files = sorted(filter_files(
            PROCESS_DIRECTORY, ('dng',)))
        test_dng_files = sorted(convert_raw_files_to_dng_files(
            RAW_IMAGES, self._temporary_directory))

        for test_dng_file, reference_dng_file in zip(
                test_dng_files, reference_dng_files):
            np.testing.assert_almost_equal(
                read_image(str(test_dng_file)),
                read_image(str(reference_dng_file)),
                decimal=7)


class TestConvertDngFilesToIntermediateFiles(unittest.TestCase):
    """
    Defines :func:`colour_hdri.process.adobe_dng.\
convert_dng_files_to_intermediate_files` definition unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_convert_dng_files_to_intermediate_files(self):
        """
        Tests :func:`colour_hdri.process.adobe_dng.\
convert_dng_files_to_intermediate_files` definition.
        """

        reference_dng_files = sorted(
            filter_files(PROCESS_DIRECTORY, ('dng',)))
        tests_dng_files = [os.path.join(self._temporary_directory,
                                        os.path.basename(reference_dng_file))
                           for reference_dng_file in reference_dng_files]
        for reference_dng_file, tests_dng_file in zip(reference_dng_files,
                                                      tests_dng_files):
            shutil.copyfile(reference_dng_file, tests_dng_file)

        reference_tiff_files = sorted(
            filter_files(PROCESS_DIRECTORY, ('tiff',)))

        test_tiff_files = sorted(convert_dng_files_to_intermediate_files(
            tests_dng_files, self._temporary_directory))

        for test_tiff_file, reference_tiff_file in zip(
                test_tiff_files, reference_tiff_files):
            np.testing.assert_almost_equal(
                read_image(str(test_tiff_file)),
                read_image(str(reference_tiff_file)),
                decimal=7)


class TestReadDngFilesExifTags(unittest.TestCase):
    """
    Defines :func:`colour_hdri.process.adobe_dng.\
read_dng_files_exif_tags` definition unit tests methods.
    """

    def test_read_dng_files_exif_tags(self):
        """
        Tests :func:`colour_hdri.process.adobe_dng.\
read_dng_files_exif_tags` definition.
        """

        reference_dng_files = sorted(
            filter_files(PROCESS_DIRECTORY, ('dng',)))
        exif_tags = read_dng_files_exif_tags(reference_dng_files)
        self.assertEqual(len(exif_tags), 3)
        self.assertIn('EXIF', exif_tags[0])
        self.assertIn('Make', exif_tags[0]['EXIF'])

        self.assertAlmostEquals(
            exif_tags[0]['EXIF']['Exposure Time'],
            0.12500000,
            places=7)

        np.testing.assert_array_equal(
            exif_tags[0]['EXIF']['Reduction Matrix 1'],
            np.identity(3))


if __name__ == '__main__':
    unittest.main()
