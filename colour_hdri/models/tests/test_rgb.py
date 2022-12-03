# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour_hdri.models.rgb` module."""

import numpy as np
import unittest

from colour_hdri.models import camera_space_to_RGB, camera_space_to_sRGB

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestCameraSpaceToRGB",
    "TestCameraSpaceTosRGB",
]


class TestCameraSpaceToRGB(unittest.TestCase):
    """
    Define :func:`colour_hdri.models.rgb.camera_space_to_RGB` definition
    unit tests methods.
    """

    def test_camera_space_to_RGB(self):
        """Test :func:`colour_hdri.models.rgb.camera_space_to_RGB` definition."""

        np.testing.assert_array_almost_equal(
            camera_space_to_RGB(
                np.array([0.80660, 0.81638, 0.65885]),
                np.array(
                    [
                        [0.47160000, 0.06030000, -0.08300000],
                        [-0.77980000, 1.54740000, 0.24800000],
                        [-0.14960000, 0.19370000, 0.66510000],
                    ]
                ),
                np.array(
                    [
                        [0.41238656, 0.35759149, 0.18045049],
                        [0.21263682, 0.71518298, 0.07218020],
                        [0.01933062, 0.11919716, 0.95037259],
                    ]
                ),
            ),
            np.array([0.75641808, 0.86831924, 0.60445896]),
            decimal=7,
        )


class TestCameraSpaceTosRGB(unittest.TestCase):
    """
    Define :func:`colour_hdri.models.rgb.camera_space_to_sRGB` definition
    unit tests methods.
    """

    def test_camera_space_to_sRGB(self):
        """Test :func:`colour_hdri.models.rgb.camera_space_to_sRGB` definition."""

        np.testing.assert_array_almost_equal(
            camera_space_to_sRGB(
                np.array([0.80660, 0.81638, 0.65885]),
                np.array(
                    [
                        [0.47160000, 0.06030000, -0.08300000],
                        [-0.77980000, 1.54740000, 0.24800000],
                        [-0.14960000, 0.19370000, 0.66510000],
                    ]
                ),
            ),
            np.array([0.75643502, 0.86831555, 0.60447061]),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
