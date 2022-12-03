# !/usr/bin/env python
"""
Define the unit tests for the :mod:`colour_hdri.models.dng` module.

Notes
-----
-   The various matrices used in this module are extracted from a
    *Canon EOS 5D Mark II* camera.
"""

from __future__ import annotations

import numpy as np
import unittest

from colour.hints import Floating, NDArray

from colour_hdri.models.dng import matrix_interpolated
from colour_hdri.models import (
    xy_to_camera_neutral,
    camera_neutral_to_xy,
    matrix_XYZ_to_camera_space,
    matrix_camera_space_to_XYZ,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CCT_CALIBRATION_ILLUMINANT_1",
    "CCT_CALIBRATION_ILLUMINANT_2",
    "M_COLOR_MATRIX_1",
    "M_COLOR_MATRIX_2",
    "M_CAMERA_CALIBRATION_1",
    "M_CAMERA_CALIBRATION_2",
    "ANALOG_BALANCE",
    "M_FORWARD_MATRIX_1",
    "M_FORWARD_MATRIX_2",
    "TestMatrixInterpolated",
    "TestXy_to_camera_neutral",
    "TestCamera_neutral_to_xy",
    "TestMatrix_XYZ_to_camera_space",
    "TestMatrix_Camera_space_to_XYZ",
]

CCT_CALIBRATION_ILLUMINANT_1: Floating = 2850
CCT_CALIBRATION_ILLUMINANT_2: Floating = 6500

M_COLOR_MATRIX_1: NDArray = np.array(
    [
        [0.5309, -0.0229, -0.0336],
        [-0.6241, 1.3265, 0.3337],
        [-0.0817, 0.1215, 0.6664],
    ]
)
M_COLOR_MATRIX_2: NDArray = np.array(
    [
        [0.4716, 0.0603, -0.0830],
        [-0.7798, 1.5474, 0.2480],
        [-0.1496, 0.1937, 0.6651],
    ]
)

M_CAMERA_CALIBRATION_1: NDArray = np.array(
    [
        [0.9603, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.9664],
    ]
)
M_CAMERA_CALIBRATION_2: NDArray = np.array(
    [
        [0.9603, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.9664],
    ]
)

ANALOG_BALANCE: NDArray = np.ones(3)

M_FORWARD_MATRIX_1: NDArray = np.array(
    [
        [0.8924, -0.1041, 0.1760],
        [0.4351, 0.6621, -0.0972],
        [0.0505, -0.1562, 0.9308],
    ]
)
M_FORWARD_MATRIX_2: NDArray = np.array(
    [
        [0.8924, -0.1041, 0.1760],
        [0.4351, 0.6621, -0.0972],
        [0.0505, -0.1562, 0.9308],
    ]
)


class TestMatrixInterpolated(unittest.TestCase):
    """
    Define :func:`colour_hdri.models.adobe_dng.matrix_interpolated` definition
    unit tests methods.
    """

    def test_matrix_interpolated(self):
        """
        Test :func:`colour_hdri.models.adobe_dng.matrix_interpolated`
        definition.
        """

        M_reference = np.array(
            [
                [0.48549082, 0.04081068, -0.07142822],
                [-0.74332781, 1.49565493, 0.26807493],
                [-0.13369466, 0.17678740, 0.66540452],
            ]
        )
        np.testing.assert_array_almost_equal(
            matrix_interpolated(
                5000, 2850, 6500, M_COLOR_MATRIX_1, M_COLOR_MATRIX_2
            ),
            M_reference,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            matrix_interpolated(
                2850, 2850, 6500, M_COLOR_MATRIX_1, M_COLOR_MATRIX_2
            ),
            M_COLOR_MATRIX_1,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            matrix_interpolated(
                1000, 2850, 6500, M_COLOR_MATRIX_1, M_COLOR_MATRIX_2
            ),
            M_COLOR_MATRIX_1,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            matrix_interpolated(
                6500, 2850, 6500, M_COLOR_MATRIX_1, M_COLOR_MATRIX_2
            ),
            M_COLOR_MATRIX_2,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            matrix_interpolated(
                10000, 2850, 6500, M_COLOR_MATRIX_1, M_COLOR_MATRIX_2
            ),
            M_COLOR_MATRIX_2,
            decimal=7,
        )


class TestXy_to_camera_neutral(unittest.TestCase):
    """
    Define :func:`colour_hdri.models.adobe_dng.\
xy_to_camera_neutral` definition unit tests methods.
    """

    def test_xy_to_camera_neutral(self):
        """
        Test :func:`colour_hdri.models.adobe_dng.\
xy_to_camera_neutral` definition.
        """

        np.testing.assert_array_almost_equal(
            xy_to_camera_neutral(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
            ),
            np.array([0.41306999, 1.00000000, 0.64646500]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            xy_to_camera_neutral(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                np.identity(3),
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
            ),
            np.array([0.42696744, 1.00000000, 0.63712786]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            xy_to_camera_neutral(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                M_CAMERA_CALIBRATION_1,
                M_CAMERA_CALIBRATION_2,
                ANALOG_BALANCE,
            ),
            np.array([0.39667111, 1.00000000, 0.62474378]),
            decimal=7,
        )


class TestCamera_neutral_to_xy(unittest.TestCase):
    """
    Define :func:`colour_hdri.models.adobe_dng.\
camera_neutral_to_xy` definition unit tests methods.
    """

    def test_camera_neutral_to_xy(self):
        """
        Test :func:`colour_hdri.models.adobe_dng.\
camera_neutral_to_xy` definition.
        """

        np.testing.assert_array_almost_equal(
            camera_neutral_to_xy(
                np.array([0.41306999, 1.00000000, 0.64646500]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
            ),
            np.array([0.32816244, 0.34698169]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            camera_neutral_to_xy(
                np.array([0.42696744, 1.00000000, 0.63712786]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                np.identity(3),
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
            ),
            np.array([0.32816244, 0.34698169]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            camera_neutral_to_xy(
                np.array([0.39667111, 1.00000000, 0.62474378]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                M_CAMERA_CALIBRATION_1,
                M_CAMERA_CALIBRATION_2,
                ANALOG_BALANCE,
            ),
            np.array([0.32816244, 0.34698169]),
            decimal=7,
        )


class TestMatrix_XYZ_to_camera_space(unittest.TestCase):
    """
    Define :func:`colour_hdri.models.adobe_dng.\
matrix_XYZ_to_camera_space` definition unit tests methods.
    """

    def test_matrix_XYZ_to_camera_space(self):
        """
        Test :func:`colour_hdri.models.adobe_dng.\
matrix_XYZ_to_camera_space` definition.
        """
        M_reference = np.array(
            [
                [0.47823951, 0.05098453, -0.07746894],
                [-0.76236708, 1.52266698, 0.25759538],
                [-0.14199759, 0.18561614, 0.66524555],
            ]
        )
        np.testing.assert_array_almost_equal(
            matrix_XYZ_to_camera_space(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
            ),
            M_reference,
            decimal=7,
        )

        M_reference = np.array(
            [
                [0.47823951, 0.05098453, -0.07746894],
                [-0.76236708, 1.52266698, 0.25759538],
                [-0.14199759, 0.18561614, 0.66524555],
            ]
        )
        np.testing.assert_array_almost_equal(
            matrix_XYZ_to_camera_space(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
            ),
            M_reference,
            decimal=7,
        )

        M_reference = np.array(
            [
                [0.47823951, 0.05098453, -0.07746894],
                [-0.76236708, 1.52266698, 0.25759538],
                [-0.14199759, 0.18561614, 0.66524555],
            ]
        )
        np.testing.assert_array_almost_equal(
            matrix_XYZ_to_camera_space(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
            ),
            M_reference,
            decimal=7,
        )


class TestMatrix_Camera_space_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour_hdri.models.adobe_dng.\
matrix_camera_space_to_XYZ` definition unit tests methods.
    """

    def test_matrix_camera_space_to_XYZ(self):
        """
        Test :func:`colour_hdri.models.adobe_dng.\
matrix_camera_space_to_XYZ` definition.
        """

        M_reference = np.array(
            [
                [2.16040871, -0.10410000, 0.27224985],
                [1.05333240, 0.66210000, -0.15035617],
                [0.12225531, -0.15620000, 1.43983046],
            ]
        )
        np.testing.assert_array_almost_equal(
            matrix_camera_space_to_XYZ(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
                M_FORWARD_MATRIX_1,
                M_FORWARD_MATRIX_2,
            ),
            M_reference,
            decimal=7,
        )

        M_reference = np.array(
            [
                [2.16040871, -0.10410000, 0.27224985],
                [1.05333240, 0.66210000, -0.15035617],
                [0.12225531, -0.15620000, 1.43983046],
            ]
        )
        np.testing.assert_array_almost_equal(
            matrix_camera_space_to_XYZ(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
                M_FORWARD_MATRIX_1,
                M_FORWARD_MATRIX_2,
            ),
            M_reference,
            decimal=7,
        )

        M_reference = np.array(
            [
                [2.16040871, -0.10410000, 0.27224985],
                [1.05333240, 0.66210000, -0.15035617],
                [0.12225531, -0.15620000, 1.43983046],
            ]
        )
        np.testing.assert_array_almost_equal(
            matrix_camera_space_to_XYZ(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
                M_FORWARD_MATRIX_1,
                M_FORWARD_MATRIX_2,
            ),
            M_reference,
            decimal=7,
        )

        M_reference = np.array(
            [
                [2.08092549, -0.08875351, 0.23804475],
                [1.00983630, 0.63301091, -0.14108066],
                [0.13472434, -0.17097714, 1.40232276],
            ]
        )
        np.testing.assert_array_almost_equal(
            matrix_camera_space_to_XYZ(
                np.array([0.32816244, 0.34698169]),
                2850,
                6500,
                M_COLOR_MATRIX_1,
                M_COLOR_MATRIX_2,
                np.identity(3),
                np.identity(3),
                ANALOG_BALANCE,
                np.identity(3),
                np.identity(3),
            ),
            M_reference,
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
