# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.utilities.common` module.
"""

from __future__ import division, unicode_literals

import os
import unittest

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.utilities import (
    vivification,
    vivified_to_dict,
    path_exists,
    filter_files)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FROBISHER_001_DIRECTORY',
           'TestVivification',
           'TestVivifiedToDict',
           'TestPathExists',
           'TestFilterFiles']

FROBISHER_001_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'frobisher_001')


class TestVivification(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.common.vivification` definition unit
    tests methods.
    """

    def test_vivification(self):
        """
        Tests :func:`colour_hdri.utilities.common.vivification` definition.
        """

        vivified = vivification()
        vivified['my']['attribute'] = 1
        self.assertIn('attribute', vivified['my'].keys())
        self.assertEqual(vivified['my']['attribute'], 1)


class TestVivifiedToDict(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.common.vivified_to_dict` definition
    unit tests methods.
    """

    def test_vivified_to_dict(self):
        """
        Tests :func:`colour_hdri.utilities.common.vivified_to_dict` definition.
        """

        vivified = vivification()
        vivified['my']['attribute'] = 1
        vivified_as_dict = vivified_to_dict(vivified)
        self.assertIsInstance(dict(), type(vivified_as_dict))
        self.assertIn('attribute', vivified_as_dict['my'].keys())
        self.assertEqual(vivified_as_dict['my']['attribute'], 1)


class TestPathExists(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.common.path_exists` definition unit
    tests methods.
    """

    def test_path_exists(self):
        """
        Tests :func:`colour_hdri.utilities.common.path_exists` definition.
        """

        self.assertTrue(path_exists(__file__))
        self.assertFalse(path_exists(''))


class TestFilterFiles(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.common.filter_files` definition unit
    tests methods.
    """

    def test_filter_files(self):
        """
        Tests :func:`colour_hdri.utilities.common.filter_files` definition.
        """

        raw_files = filter_files(FROBISHER_001_DIRECTORY, ('CR2', 'jpg'))
        self.assertListEqual(sorted(map(os.path.basename, raw_files)),
                             ['IMG_2600.CR2', 'IMG_2600.jpg',
                              'IMG_2601.CR2', 'IMG_2601.jpg',
                              'IMG_2602.CR2', 'IMG_2602.jpg'])


if __name__ == '__main__':
    unittest.main()
