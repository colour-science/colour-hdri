# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour_hdri.exposure.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour_hdri.exposure import (
    average_luminance, average_illuminance, luminance_to_exposure_value,
    illuminance_to_exposure_value, adjust_exposure)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestAverageLuminance', 'TestAverageIlluminance',
    'TestLuminanceToExposureValue', 'TestAdjustExposure'
]


class TestAverageLuminance(unittest.TestCase):
    """
    Defines :func:`colour_hdri.exposure.common.average_luminance` definition
    unit tests methods.
    """

    def test_average_luminance(self):
        """
        Tests :func:`colour_hdri.exposure.common.average_luminance` definition.
        """

        np.testing.assert_almost_equal(
            average_luminance(
                np.array([2.8, 5.6, 8]),
                np.array([0.125, 0.5, 1.0]),
                np.array([100, 800, 16000]),
            ),
            np.array([7.84000000, 0.98000000, 0.05000000]),
            decimal=7)


class TestAverageIlluminance(unittest.TestCase):
    """
    Defines :func:`colour_hdri.exposure.common.average_illuminance` definition
    unit tests methods.
    """

    def test_average_illuminance(self):
        """
        Tests :func:`colour_hdri.exposure.common.average_illuminance`
        definition.
        """

        np.testing.assert_almost_equal(
            average_illuminance(
                np.array([2.8, 5.6, 8]),
                np.array([0.125, 0.5, 1.0]),
                np.array([100, 800, 16000]),
            ),
            np.array([156.80000000, 19.60000000, 1.00000000]),
            decimal=7)


class TestLuminanceToExposureValue(unittest.TestCase):
    """
    Defines :func:`colour_hdri.exposure.common.luminance_to_exposure_value`
    definition unit tests methods.
    """

    def test_luminance_to_exposure_value(self):
        """
        Tests :func:`colour_hdri.exposure.common.luminance_to_exposure_value`
        definition.
        """

        np.testing.assert_almost_equal(
            luminance_to_exposure_value(
                np.array([0.125, 0.250, 0.125]),
                np.array([100, 100, 100]),
                np.array([12.5, 12.5, 14]),
            ),
            np.array([0.00000000, 1.00000000, -0.16349873]),
            decimal=7)


class TestIlluminanceToExposureValue(unittest.TestCase):
    """
    Defines :func:`colour_hdri.exposure.common.illuminance_to_exposure_value`
    definition unit tests methods.
    """

    def test_illuminance_to_exposure_value(self):
        """
        Tests :func:`colour_hdri.exposure.common.illuminance_to_exposure_value`
        definition.
        """

        np.testing.assert_almost_equal(
            illuminance_to_exposure_value(
                np.array([2.5, 5.0, 0.125]),
                np.array([100, 100, 100]),
                np.array([250, 250, 340]),
            ),
            np.array([0.00000000, 1.00000000, -4.76553475]),
            decimal=7)


class TestAdjustExposure(unittest.TestCase):
    """
    Defines :func:`colour_hdri.exposure.common.adjust_exposure` definition
    unit tests methods.
    """

    def test_adjust_exposure(self):
        """
        Tests :func:`colour_hdri.exposure.common.adjust_exposure` definition.
        """

        np.testing.assert_almost_equal(
            adjust_exposure(np.array([0.25, 0.5, 0.75, 1]), 1),
            np.array([0.5, 1.0, 1.5, 2.]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
