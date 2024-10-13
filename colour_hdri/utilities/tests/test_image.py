"""Define the unit tests for the :mod:`colour_hdri.utilities.image` module."""

from __future__ import annotations

import os

import numpy as np
from colour.constants import TOLERANCE_ABSOLUTE_TESTS

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.utilities import Image, ImageStack, filter_files

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestImage",
    "TestImageStack",
]

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(ROOT_RESOURCES_TESTS, "frobisher_001")


class TestImage:
    """
    Define :class:`colour_hdri.utilities.image.Image` class unit tests
    methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._test_jpg_image = filter_files(ROOT_RESOURCES_FROBISHER_001, ("jpg",))[0]

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("path", "data", "metadata")

        for attribute in required_attributes:
            assert attribute in dir(Image)

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__init__", "read_data", "read_metadata")

        for method in required_methods:
            assert method in dir(Image)

    def test_read_data(self):
        """Test :attr:`colour_hdri.utilities.image.Image.read_data` method."""

        image = Image(self._test_jpg_image)

        assert image.data is None
        assert image.read_data().shape == (426, 640, 3)

    def test_read_metadata(self):
        """Test :attr:`colour_hdri.utilities.image.Image.end` method."""

        image = Image(self._test_jpg_image)

        assert image.metadata is None
        np.testing.assert_array_equal(
            np.array(image.read_metadata()),
            np.array([8.0, 0.125, 100.0, np.nan, np.nan, np.nan]),
        )


class TestImageStack:
    """
    Define :class:`colour_hdri.utilities.image.ImageStack` class unit tests
    methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._test_jpg_images = filter_files(ROOT_RESOURCES_FROBISHER_001, ("jpg",))

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
            assert method in dir(ImageStack)

    def test__getitem__(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.__getitem__`
        method.
        """

        for image in self._image_stack:
            assert isinstance(image, Image)

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

        assert image_stack[0] == image

    def test__delitem__(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.__delitem__`
        method.
        """

        image_stack = ImageStack().from_files(self._test_jpg_images)

        del image_stack[0]

        assert len(image_stack) == 2

    def test__len__(self):
        """Test :attr:`colour_hdri.utilities.image.ImageStack.__len__` method."""

        assert len(self._image_stack) == 3

    def test__getattr__(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.__getattr__`
        method.
        """

        assert self._image_stack.data.shape == (426, 640, 3, 3)

        np.testing.assert_allclose(
            self._image_stack.f_number,
            np.array([8, 8, 8]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert self._image_stack[0].metadata.f_number == 8

        np.testing.assert_allclose(
            self._image_stack.exposure_time,
            np.array([0.125, 1, 8]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert self._image_stack[0].metadata.exposure_time == 0.125

        np.testing.assert_array_equal(
            self._image_stack.black_level, np.array([np.nan, np.nan, np.nan])
        )

        assert self._image_stack[0].metadata.black_level is None

    def test__setattr__(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.__getattr__`
        method.
        """

        image_stack = ImageStack().from_files(self._test_jpg_images)

        assert image_stack.data.shape == (426, 640, 3, 3)

        image_stack.data = np.random.random((20, 10, 3, 3))

        assert image_stack.data.shape == (20, 10, 3, 3)

        np.testing.assert_allclose(
            image_stack.f_number,
            np.array([8, 8, 8]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        image_stack.f_number = np.array([1, 2, 3])

        np.testing.assert_allclose(
            image_stack.f_number,
            np.array([1, 2, 3]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert image_stack[0].metadata.f_number == 1

        np.testing.assert_array_equal(
            image_stack.black_level, np.array([np.nan, np.nan, np.nan])
        )

        image_stack.black_level = np.array([2048, 2048, 2048])

        np.testing.assert_allclose(
            image_stack.black_level,
            np.array([2048, 2048, 2048]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert image_stack[0].metadata.black_level == 2048

    def test_from_files(self):
        """
        Test :attr:`colour_hdri.utilities.image.ImageStack.test_from_files`
        method.
        """

        image_stack = ImageStack().from_files(reversed(self._test_jpg_images))
        assert list(image_stack.path) == self._test_jpg_images
