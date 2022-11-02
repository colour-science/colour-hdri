# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour_hdri.utilities.image` module."""

from __future__ import annotations

import numpy as np
import os
import unittest

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.utilities import filter_files
from colour_hdri.utilities import Image, ImageStack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestImage",
    "TestImageStack",
]

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(
    ROOT_RESOURCES_TESTS, "frobisher_001"
)


class TestImage(unittest.TestCase):
    """
    Define :class:`colour_hdri.utilities.image.Image` class unit tests
    methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._test_jpg_image = filter_files(
            ROOT_RESOURCES_FROBISHER_001, ("jpg",)
        )[0]

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("path", "data", "metadata")

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Image))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__init__", "read_data", "read_metadata")

        for method in required_methods:
            self.assertIn(method, dir(Image))

    def test_read_data(self):
        """Test :attr:`colour_hdri.utilities.image.Image.read_data` method."""

        image = Image(self._test_jpg_image)

        self.assertIsNone(image.data)
        self.assertTupleEqual(image.read_data().shape, (426, 640, 3))

    def test_read_metadata(self):
        """Test :attr:`colour_hdri.utilities.image.Image.end` method."""

        image = Image(self._test_jpg_image)

        self.assertEqual(image.metadata, None)
        np.testing.assert_array_equal(
            np.array(image.read_metadata()),
            np.array([8.0, 0.125, 100.0, np.nan, np.nan, np.nan]),
        )


class TestImageStack(unittest.TestCase):
    """
    Define :class:`colour_hdri.utilities.image.ImageStack` class unit tests
    methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._test_jpg_images = filter_files(
            ROOT_RESOURCES_FROBISHER_001, ("jpg",)
        )

        self._image_stack = ImageStack().from_files(self._test_jpg_images)

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__getitem__",
            "__setitem__",
            "__delitem__",
            "__len__",
            "__getattr__",
            "__setattr__",
            "insert",
            "from_files",
        )

        for method in required_methods:
            self.assertIn(method, dir(ImageStack))

    def test__getitem__(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.__getitem__`
        method.
        """

        for image in self._image_stack:
            self.assertIsInstance(image, Image)

    def test__setitem__(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.__setitem__`
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
        Test :attr:`colour_hdri.utilities.image.ImageStack.__delitem__`
        method.
        """

        image_stack = ImageStack().from_files(self._test_jpg_images)

        del image_stack[0]

        self.assertEqual(len(image_stack), 2)

    def test__len__(self):
        """Test :attr:`colour_hdri.utilities.image.ImageStack.__len__` method."""

        self.assertEqual(len(self._image_stack), 3)

    def test__getattr__(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.__getattr__`
        method.
        """

        self.assertTupleEqual(self._image_stack.data.shape, (426, 640, 3, 3))

        np.testing.assert_array_almost_equal(
            self._image_stack.f_number, np.array([8, 8, 8]), decimal=7
        )

        self.assertEqual(self._image_stack[0].metadata.f_number, 8)

        np.testing.assert_array_almost_equal(
            self._image_stack.exposure_time, np.array([0.125, 1, 8]), decimal=7
        )

        self.assertEqual(self._image_stack[0].metadata.exposure_time, 0.125)

        np.testing.assert_array_equal(
            self._image_stack.black_level, np.array([np.nan, np.nan, np.nan])
        )

        self.assertEqual(self._image_stack[0].metadata.black_level, None)

    def test__setattr__(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.__getattr__`
        method.
        """

        image_stack = ImageStack().from_files(self._test_jpg_images)

        self.assertTupleEqual(image_stack.data.shape, (426, 640, 3, 3))

        image_stack.data = np.random.random((20, 10, 3, 3))

        self.assertTupleEqual(image_stack.data.shape, (20, 10, 3, 3))

        np.testing.assert_array_almost_equal(
            image_stack.f_number, np.array([8, 8, 8]), decimal=7
        )

        image_stack.f_number = np.array([1, 2, 3])

        np.testing.assert_array_almost_equal(
            image_stack.f_number, np.array([1, 2, 3]), decimal=7
        )

        self.assertEqual(image_stack[0].metadata.f_number, 1)

        np.testing.assert_array_equal(
            image_stack.black_level, np.array([np.nan, np.nan, np.nan])
        )

        image_stack.black_level = np.array([2048, 2048, 2048])

        np.testing.assert_array_almost_equal(
            image_stack.black_level, np.array([2048, 2048, 2048]), decimal=7
        )

        self.assertEqual(image_stack[0].metadata.black_level, 2048)

    def test_from_files(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.test_from_files`
        method.
        """

        image_stack = ImageStack().from_files(reversed(self._test_jpg_images))
        self.assertListEqual(list(image_stack.path), self._test_jpg_images)


if __name__ == "__main__":
    unittest.main()
