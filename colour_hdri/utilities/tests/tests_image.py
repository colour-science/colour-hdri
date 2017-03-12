# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.utilities.image` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import unittest

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.utilities import filter_files
from colour_hdri.utilities import Metadata, Image, ImageStack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestMetadata']

FROBISHER_001_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'frobisher_001')


class TestMetadata(unittest.TestCase):
    """
    Defines :class:`colour_hdri.utilities.image.Metadata` class unit tests
    methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('f_number',
                               'exposure_time',
                               'iso',
                               'black_level',
                               'white_level',
                               'white_balance_multipliers')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Metadata))


class TestImage(unittest.TestCase):
    """
    Defines :class:`colour_hdri.utilities.image.Image` class unit tests
    methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._test_jpg_image = filter_files(
            FROBISHER_001_DIRECTORY, ('jpg',))[0]

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('path',
                               'data',
                               'metadata')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Image))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('read_data',
                            'read_metadata')

        for method in required_methods:
            self.assertIn(method, dir(Image))

    def test_read_data(self):
        """
        Tests :attr:`colour_hdri.utilities.image.Image.read_data` method.
        """

        image = Image(self._test_jpg_image)

        self.assertEqual(image.data, np.array(None))
        self.assertTupleEqual(image.read_data().shape, (426, 640, 3))

    def test_read_metadata(self):
        """
        Tests :attr:`colour_hdri.utilities.image.Image.end` method.
        """

        image = Image(self._test_jpg_image)

        self.assertEqual(image.metadata, None)
        self.assertTupleEqual(tuple(image.read_metadata()),
                              (8.0, 0.125, 100.0, None, None, None))


class TestImageStack(unittest.TestCase):
    """
    Defines :class:`colour_hdri.utilities.image.ImageStack` class unit tests
    methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._test_jpg_images = filter_files(
            FROBISHER_001_DIRECTORY, ('jpg',))

        self._image_stack = ImageStack().from_files(self._test_jpg_images)

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = (
            '__init__',
            '__getitem__',
            '__setitem__',
            '__delitem__',
            '__len__',
            '__getattr__',
            '__setattr__',
            'insert',
            'from_files')

        for method in required_methods:
            self.assertIn(method, dir(ImageStack))

    def test__getitem__(self):
        """
        Tests :attr:`colour_hdri.utilities.image.ImageStack.__getitem__`
        method.
        """

        for image in self._image_stack:
            self.assertIsInstance(image, Image)

    def test__setitem__(self):
        """
        Tests :attr:`colour_hdri.utilities.image.ImageStack.__setitem__`
        method.
        """

        image_stack = ImageStack()
        image = Image(self._test_jpg_images[0])
        image.read_data()
        image.read_metadata()
        image_stack.insert(0, image)

        self.assertEqual(image_stack[0], image)

    def test__delitem__(self):
        """
        Tests :attr:`colour_hdri.utilities.image.ImageStack.__delitem__`
        method.
        """

        image_stack = ImageStack().from_files(self._test_jpg_images)

        del image_stack[0]

        self.assertEqual(len(image_stack), 2)

    def test__len__(self):
        """
        Tests :attr:`colour_hdri.utilities.image.ImageStack.__len__` method.
        """

        self.assertEqual(len(self._image_stack), 3)

    def test__getattr__(self):
        """
        Tests :attr:`colour_hdri.utilities.image.ImageStack.__getattr__`
        method.
        """

        self.assertTupleEqual(
            self._image_stack.data.shape,
            (426, 640, 3, 3))

        np.testing.assert_almost_equal(
            self._image_stack.f_number,
            np.array([8, 8, 8]),
            decimal=7)

        self.assertEqual(self._image_stack[0].metadata.f_number, 8)

        np.testing.assert_almost_equal(
            self._image_stack.exposure_time,
            np.array([0.125, 1, 8]),
            decimal=7)

        self.assertEqual(self._image_stack[0].metadata.exposure_time, 0.125)

        self.assertListEqual(
            list(self._image_stack.black_level),
            [None, None, None])

        self.assertEqual(self._image_stack[0].metadata.black_level, None)

    def test__setattr__(self):
        """
        Tests :attr:`colour_hdri.utilities.image.ImageStack.__getattr__`
        method.
        """

        image_stack = ImageStack().from_files(self._test_jpg_images)

        self.assertTupleEqual(
            image_stack.data.shape,
            (426, 640, 3, 3))

        image_stack.data = np.random.random((20, 10, 3, 3))

        self.assertTupleEqual(
            image_stack.data.shape,
            (20, 10, 3, 3))

        np.testing.assert_almost_equal(
            image_stack.f_number,
            np.array([8, 8, 8]),
            decimal=7)

        image_stack.f_number = np.array([1, 2, 3])

        np.testing.assert_almost_equal(
            image_stack.f_number,
            np.array([1, 2, 3]),
            decimal=7)

        self.assertEqual(image_stack[0].metadata.f_number, 1)

        self.assertListEqual(
            list(image_stack.black_level),
            [None, None, None])

        image_stack.black_level = np.array([2048, 2048, 2048])

        np.testing.assert_almost_equal(
            image_stack.black_level,
            np.array([2048, 2048, 2048]),
            decimal=7)

        self.assertEqual(image_stack[0].metadata.black_level, 2048)


if __name__ == '__main__':
    unittest.main()
