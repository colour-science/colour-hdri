# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.utilities.exif` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import shutil
import tempfile
import unittest

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.utilities import filter_files, vivified_to_dict
from colour_hdri.utilities import (
    ExifTag,
    parse_exif_string,
    parse_exif_numeric,
    parse_exif_fraction,
    parse_exif_array,
    parse_exif_data,
    read_exif_tags,
    copy_exif_tags,
    update_exif_tags,
    delete_exif_tags,
    read_exif_tag,
    write_exif_tag)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FROBISHER_001_DIRECTORY',
           'TestParseExifString',
           'TestParseExifNumeric',
           'TestParseExifFraction',
           'TestParseExifArray',
           'TestParseExifData',
           'TestReadExifTags',
           'TestCopyExifTags',
           'TestUpdateExifTags',
           'TestDeleteExifTags',
           'TestReadExifTag',
           'TestWriteExifTag']

FROBISHER_001_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'frobisher_001')


class TestParseExifString(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.parse_exif_string` definition
    unit tests methods.
    """

    def test_parse_exif_string(self):
        """
        Tests :func:`colour_hdri.utilities.exif.parse_exif_string` definition.
        """

        exif_tag = ExifTag('EXIF', 'Make', 'Canon', '271')
        self.assertEqual(parse_exif_string(exif_tag), 'Canon')


class TestParseExifNumeric(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.parse_exif_numeric` definition
    unit tests methods.
    """

    def test_parse_exif_numeric(self):
        """
        Tests :func:`colour_hdri.utilities.exif.parse_exif_numeric` definition.
        """

        exif_tag = ExifTag('EXIF', 'Focal Length', '16', '37386')
        self.assertEqual(parse_exif_numeric(exif_tag), 16)

        exif_tag = ExifTag('EXIF', 'Focal Length', '16', '37386')
        self.assertIsInstance(parse_exif_numeric(exif_tag, np.int_), np.int_)


class TestParseExifFraction(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.parse_exif_fraction` definition
    unit tests methods.
    """

    def test_parse_exif_fraction(self):
        """
        Tests :func:`colour_hdri.utilities.exif.parse_exif_fraction`
        definition.
        """

        exif_tag = ExifTag('EXIF', 'Exposure Time', '0.01666666667', '33434')
        self.assertAlmostEqual(
            parse_exif_fraction(exif_tag),
            0.01666666,
            places=7)

        exif_tag = ExifTag('EXIF', 'Exposure Time', '10/4000', '33434')
        self.assertAlmostEqual(
            parse_exif_fraction(exif_tag),
            0.00250000,
            places=7)

        self.assertIsInstance(parse_exif_fraction(exif_tag, np.int_), np.int_)


class TestParseExifArray(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.parse_exif_array` definition
    unit tests methods.
    """

    def test_parse_exif_array(self):
        """
        Tests :func:`colour_hdri.utilities.exif.parse_exif_array` definition.
        """

        exif_tag = ExifTag('EXIF',
                           'Color Matrix 1',
                           ('0.5309 -0.0229 -0.0336 '
                            '-0.6241 1.3265 0.3337 '
                            '-0.0817 0.1215 0.6664'),
                           '50721')
        np.testing.assert_array_equal(
            parse_exif_array(exif_tag),
            np.array([
                0.5309, -0.0229, -0.0336,
                -0.6241, 1.3265, 0.3337,
                -0.0817, 0.1215, 0.6664]))

        np.testing.assert_array_equal(
            parse_exif_array(exif_tag, shape=(3, 3)),
            np.array([
                [0.5309, -0.0229, -0.0336],
                [-0.6241, 1.3265, 0.3337],
                [-0.0817, 0.1215, 0.6664]]))


class TestParseExifData(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.parse_exif_data` definition
    unit tests methods.
    """

    def test_parse_exif_data(self):
        """
        Tests :func:`colour_hdri.utilities.exif.parse_exif_data` definition.
        """

        self.assertListEqual(
            parse_exif_data(
                '[XMP]               - Description                     :'),
            ['XMP', '-', 'Description', ''])

        self.assertListEqual(
            parse_exif_data(
                '[EXIF]            296 Resolution Unit                 : 2'),
            ['EXIF', '296', 'Resolution Unit', '2'])

        self.assertListEqual(
            parse_exif_data(
                '[ICC_Profile]       8 Profile Version                 : 528'),
            ['ICC_Profile', '8', 'Profile Version', '528'])


class TestReadExifTags(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.read_exif_tags` definition unit
    tests methods.
    """

    def test_read_exif_tags(self):
        """
        Tests :func:`colour_hdri.utilities.exif.read_exif_tags` definition.
        """

        test_jpg_image = filter_files(FROBISHER_001_DIRECTORY, ('jpg',))[0]
        exif_data = vivified_to_dict(read_exif_tags(test_jpg_image))

        self.assertIsInstance(exif_data, type(dict()))

        self.assertListEqual(
            sorted(exif_data.keys()),
            ['Composite', 'EXIF', 'ExifTool', 'File', 'ICC_Profile', 'JFIF',
             'Photoshop', 'XMP'])

        self.assertListEqual(
            sorted(exif_data['EXIF'].values()),
            [[ExifTag('EXIF', 'Camera Model Name', 'EOS 5D Mark II', '272')],
             [ExifTag('EXIF', 'Create Date', '2015:09:19 03:39:20', '36868')],
             [ExifTag('EXIF', 'Date/Time Original',
                      '2015:09:19 03:39:20', '36867')],
             [ExifTag('EXIF', 'Exif Image Height', '426', '40963')],
             [ExifTag('EXIF', 'Exif Image Width', '640', '40962')],
             [ExifTag('EXIF', 'Exposure Time', '0.125', '33434')],
             [ExifTag('EXIF', 'F Number', '8', '33437')],
             [ExifTag('EXIF', 'Focal Length', '16', '37386')],
             [ExifTag('EXIF', 'ISO', '100', '34855')],
             [ExifTag('EXIF', 'Make', 'Canon', '271')],
             [ExifTag('EXIF', 'Modify Date', '2015:09:19 03:39:20', '306')],
             [ExifTag('EXIF', 'Orientation', '1', '274')],
             [ExifTag('EXIF', 'Photometric Interpretation', '2', '262')],
             [ExifTag('EXIF', 'Resolution Unit', '2', '296')],
             [ExifTag('EXIF', 'Software', 'Photos 1.0.1', '305')],
             [ExifTag('EXIF', 'X Resolution', '72', '282')],
             [ExifTag('EXIF', 'Y Resolution', '72', '283')]])


class TestCopyExifTags(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.copy_exif_tags` definition unit
    tests methods.
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

    def test_copy_exif_tags(self):
        """
        Tests :func:`colour_hdri.utilities.exif.copy_exif_tags` definition.
        """

        reference_jpg_image = filter_files(
            FROBISHER_001_DIRECTORY, ('jpg',))[0]
        test_jpg_image = os.path.join(
            self._temporary_directory, os.path.basename(reference_jpg_image))

        shutil.copyfile(reference_jpg_image, test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, 'Aperture'), '8.0')
        delete_exif_tags(test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, 'Aperture'), '')
        copy_exif_tags(reference_jpg_image, test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, 'Aperture'), '8.0')


class TestUpdateExifTags(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.update_exif_tags` definition unit
    tests methods.
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

    def test_update_exif_tags(self):
        """
        Tests :func:`colour_hdri.utilities.exif.update_exif_tags` definition.
        """

        reference_jpg_images = filter_files(FROBISHER_001_DIRECTORY, ('jpg',))
        test_jpg_images = []
        for reference_jpg_image in reference_jpg_images:
            test_jpg_image = os.path.join(
                self._temporary_directory,
                os.path.basename(reference_jpg_image))
            shutil.copyfile(reference_jpg_image, test_jpg_image)
            delete_exif_tags(test_jpg_image)
            self.assertEqual(read_exif_tag(test_jpg_image, 'Aperture'), '')
            test_jpg_images.append(test_jpg_image)

        update_exif_tags(zip(reference_jpg_images, test_jpg_images))
        for test_jpg_image in test_jpg_images:
            self.assertEqual(read_exif_tag(test_jpg_image, 'Aperture'), '8.0')


class TestDeleteExifTags(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.delete_exif_tags` definition unit
    tests methods.
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

    def test_delete_exif_tags(self):
        """
        Tests :func:`colour_hdri.utilities.exif.delete_exif_tags` definition.
        """

        reference_jpg_image = filter_files(
            FROBISHER_001_DIRECTORY, ('jpg',))[0]
        test_jpg_image = os.path.join(
            self._temporary_directory, os.path.basename(reference_jpg_image))

        shutil.copyfile(reference_jpg_image, test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, 'Aperture'), '8.0')
        delete_exif_tags(test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, 'Aperture'), '')


class TestReadExifTag(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.read_exif_tag` definition unit
    tests methods.
    """

    def test_read_exif_tag(self):
        """
        Tests :func:`colour_hdri.utilities.exif.read_exif_tag` definition.
        """

        test_jpg_image = filter_files(FROBISHER_001_DIRECTORY, ('jpg',))[0]

        self.assertEqual(read_exif_tag(test_jpg_image, 'Aperture'), '8.0')
        self.assertEqual(read_exif_tag(test_jpg_image, 'ExposureTime'), '1/8')
        self.assertEqual(read_exif_tag(test_jpg_image, 'ISO'), '100')


class TestWriteExifTag(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.write_exif_tag` definition unit
    tests methods.
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

    def test_write_exif_tag(self):
        """
        Tests :func:`colour_hdri.utilities.exif.write_exif_tag` definition.
        """

        reference_jpg_image = filter_files(
            FROBISHER_001_DIRECTORY, ('jpg',))[0]
        test_jpg_image = os.path.join(
            self._temporary_directory, os.path.basename(reference_jpg_image))

        shutil.copyfile(reference_jpg_image, test_jpg_image)
        # *Aperture* exif tag is not writeable, changing for *FNumber*.
        self.assertEqual(read_exif_tag(test_jpg_image, 'FNumber'), '8.0')
        write_exif_tag(test_jpg_image, 'FNumber', '16.0')
        self.assertEqual(read_exif_tag(test_jpg_image, 'FNumber'), '16.0')


if __name__ == '__main__':
    unittest.main()
