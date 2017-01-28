# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.calibration.absolute_luminance`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import unittest

from colour import read_image

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.calibration import (
    upper_hemisphere_illuminance_weights_Lagarde2016,
    absolute_luminance_calibration_Lagarde2016)
from colour_hdri.calibration.absolute_luminance import (
    upper_hemisphere_illuminance_Lagarde2016)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestUpperHemisphereIlluminanceLagarde2016',
           'TestUpperHemisphereIlluminanceWeightsLagarde2016',
           'TestAbsoluteLuminanceCalibrationLagarde2016']

UNITY_001_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'unity_001')

CALIBRATION_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'colour_hdri', 'calibration')


class TestUpperHemisphereIlluminanceLagarde2016(unittest.TestCase):
    """
    Defines :func:`colour_hdri.calibration.absolute_luminance.\
upper_hemisphere_illuminance_Lagarde2016` definition unit tests methods.
    """

    def test_upper_hemisphere_illuminance_Lagarde2016(self):
        """
        Tests :func:`colour_hdri.calibration.absolute_luminance.\
upper_hemisphere_illuminance_Lagarde2016` definition.
        """

        self.assertAlmostEqual(
            upper_hemisphere_illuminance_Lagarde2016(
                np.ones((16, 32, 3))),
            2.934469165342606,
            places=7)

        self.assertAlmostEqual(
            upper_hemisphere_illuminance_Lagarde2016(
                np.ones((16, 32, 3)) * 10),
            29.344691653426061,
            places=7)

        self.assertAlmostEqual(
            upper_hemisphere_illuminance_Lagarde2016(
                np.ones((16, 32, 3)) * 0.1),
            0.293446916534261,
            places=7)


class TestUpperHemisphereIlluminanceWeightsLagarde2016(unittest.TestCase):
    """
    Defines :func:`colour_hdri.calibration.absolute_luminance.\
upper_hemisphere_illuminance_weights_Lagarde2016` definition unit tests
    methods.
    """

    def test_upper_hemisphere_illuminance_weights_Lagarde2016(self):
        """
        Tests :func:`colour_hdri.calibration.absolute_luminance.\
upper_hemisphere_illuminance_weights_Lagarde2016` definition.
        """

        weights = np.array(
            [[0.00000000],
             [1.98673676],
             [3.89213628],
             [5.63819129],
             [7.15341808],
             [8.37578310],
             [9.25524257],
             [9.75579132],
             [9.85693683],
             [9.55453819],
             [8.86097564],
             [7.80464369],
             [6.42878872],
             [4.78973839],
             [2.95459560],
             [0.99849132],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000],
             [0.00000000]])

        np.testing.assert_almost_equal(
            upper_hemisphere_illuminance_weights_Lagarde2016(32, 16),
            np.tile(weights, (1, 16)),
            decimal=7)


class TestAbsoluteLuminanceCalibrationLagarde2016(unittest.TestCase):
    """
    Defines :func:`colour_hdri.calibration.absolute_luminance.\
absolute_luminance_calibration_Lagarde2016` definition unit tests methods.
    """

    def test_absolute_luminance_calibration_Lagarde2016(self):
        """
        Tests :func:`colour_hdri.calibration.absolute_luminance.\
absolute_luminance_calibration_Lagarde2016` definition.
        """

        # Unity Technologies. (2016). Treasure Island - white balanced.exr.
        # Retrieved August 30, 2016, from http://blog.selfshadow.com/\
        # publications/s2016-shading-course/unity/supplemental/\
        # Treasure Island - white balanced.exr

        reference_exr_file = read_image(
            str(os.path.join(
                UNITY_001_DIRECTORY,
                'Unity_Treasure_Island_White_Balanced.exr')))

        test_exr_file = read_image(
            str(os.path.join(
                CALIBRATION_DIRECTORY,
                'Unity_Treasure_Island_White_Balanced_Absolute.exr')))

        np.testing.assert_allclose(
            absolute_luminance_calibration_Lagarde2016(
                reference_exr_file, 51000),
            test_exr_file,
            rtol=0.0000001,
            atol=0.0000001)


if __name__ == '__main__':
    unittest.main()
