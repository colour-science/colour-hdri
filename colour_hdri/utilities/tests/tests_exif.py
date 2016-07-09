# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.utilities.exif` module.
"""

from __future__ import division, unicode_literals
import os
import shutil
import tempfile
import unittest

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.utilities import filter_files, vivified_to_dict
from colour_hdri.utilities import (
    parse_exif_data,
    read_exif_tags,
    copy_exif_tags,
    update_exif_tags,
    delete_exif_tags,
    read_exif_tag,
    write_exif_tag)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FROBISHER_001_DIRECTORY',
           'TestParseExifData',
           'TestReadExifTags',
           'TestCopyExifTags',
           'TestUpdateExifTags',
           'TestDeleteExifTags',
           'TestReadExifTag',
           'TestWriteExifTag']

FROBISHER_001_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'frobisher_001')


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
            [('0.125', '33434'), ('1', '274'), ('100', '34855'),
             ('16', '37386'), ('2', '262'), ('2', '296'),
             ('2015:09:19 03:39:20', '306'),
             ('2015:09:19 03:39:20', '36867'),
             ('2015:09:19 03:39:20', '36868'), ('426', '40963'),
             ('640', '40962'), ('72', '282'), ('72', '283'),
             ('8', '33437'), ('Canon', '271'), ('EOS 5D Mark II', '272'),
             ('Photos 1.0.1', '305')])


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
