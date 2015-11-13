# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.utilities.exif. module.
"""

from __future__ import division, unicode_literals

import os
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.utilities import (
    filter_files,
    vivified_to_dict,
    parse_exif_data,
    read_exif_data,
    get_value,
    set_value,
    copy_tags,
    delete_all_tags,
    delete_backup_files,
    update_exif_data)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FROBISHER_001_DIRECTORY',
           'TestParseExifData',
           'TestReadExifData']

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


class TestReadExifData(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.exif.read_exif_data` definition
    unit tests methods.
    """

    def test_read_exif_data(self):
        """
        Tests :func:`colour_hdri.utilities.exif.read_exif_data` definition.
        """

        jpg_image = filter_files(FROBISHER_001_DIRECTORY, ('jpg',))[0]
        exif_data = vivified_to_dict(read_exif_data(jpg_image))

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


if __name__ == '__main__':
    unittest.main()
