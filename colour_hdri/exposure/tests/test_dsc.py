# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour_hdri.exposure.dsc` module."""

import numpy as np
import unittest

from colour_hdri.exposure.dsc import q_factor
from colour_hdri.exposure import (
    focal_plane_exposure,
    arithmetic_mean_focal_plane_exposure,
    saturation_based_speed_focal_plane_exposure,
    exposure_index_values,
    exposure_value_100,
    photometric_exposure_scale_factor_Lagarde2014,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestQFactor",
    "TestQFactor",
    "TestFocalPlaneExposure",
    "TestArithmeticMeanFocalPlaneExposure",
    "TestSaturationBasedSpeedFocalPlaneExposure",
    "TestExposureValue100",
    "TestPhotometricExposureScaleFactorLagarde2014",
]


class TestQFactor(unittest.TestCase):
    """
    Define :func:`colour_hdri.exposure.dsc.q_factor` definition
    unit tests methods.
    """

    def test_q_factor(self):
        """Test :func:`colour_hdri.exposure.dsc.q_factor` definition."""

        np.testing.assert_array_almost_equal(
            q_factor(
                np.array([9 / 10, 8 / 10, 9 / 10]),
                np.array([98 / 100, 98 / 100, 94 / 100]),
                np.array([10, 10, 20]),
            ),
            np.array([0.65157483, 0.57917763, 0.51808797]),
            decimal=7,
        )


class TestFocalPlaneExposure(unittest.TestCase):
    """
    Define :func:`colour_hdri.exposure.dsc.focal_plane_exposure` definition
    unit tests methods.
    """

    def test_focal_plane_exposure(self):
        """Test :func:`colour_hdri.exposure.dsc.focal_plane_exposure` definition."""

        np.testing.assert_array_almost_equal(
            focal_plane_exposure(
                np.array([4000, 2000, 1000]),
                np.array([8, 5.6, 2.8]),
                np.array([1 / 250, 1 / 500, 1 / 1000]),
                np.array([50 / 1000, 100 / 1000, 500 / 1000]),
                np.array([50 / 1000, 100 / 1000, 500 / 1000]),
                np.array([0.0015, 0.0050, 0.0100]),
                9 / 10,
                98 / 100,
                10,
            ),
            np.array([0.16439371, 0.08810904, 0.09310904]),
            decimal=7,
        )


class TestArithmeticMeanFocalPlaneExposure(unittest.TestCase):
    """
    Define :func:`colour_hdri.exposure.dsc.\
arithmetic_mean_focal_plane_exposure` definition unit tests methods.
    """

    def test_arithmetic_mean_focal_plane_exposure(self):
        """
        Test :func:`colour_hdri.exposure.dsc.\
arithmetic_mean_focal_plane_exposure` definition.
        """

        np.testing.assert_array_almost_equal(
            arithmetic_mean_focal_plane_exposure(
                np.array([4000, 2000, 1000]),
                np.array([8, 5.6, 2.8]),
                np.array([1 / 250, 1 / 500, 1 / 1000]),
            ),
            np.array([0.16289371, 0.08310904, 0.08310904]),
            decimal=7,
        )


class TestSaturationBasedSpeedFocalPlaneExposure(unittest.TestCase):
    """
    Define :func:`colour_hdri.exposure.dsc.\
saturation_based_speed_focal_plane_exposure` definition unit tests methods.
    """

    def test_saturation_based_speed_focal_plane_exposure(self):
        """
        Test :func:`colour_hdri.exposure.dsc.\
saturation_based_speed_focal_plane_exposure` definition.
        """

        np.testing.assert_array_almost_equal(
            saturation_based_speed_focal_plane_exposure(
                np.array([4000, 2000, 1000]),
                np.array([8, 5.6, 2.8]),
                np.array([1 / 250, 1 / 500, 1 / 1000]),
                np.array([100, 800, 1600]),
                np.array([50 / 1000, 100 / 1000, 500 / 1000]),
                np.array([50 / 1000, 100 / 1000, 500 / 1000]),
                np.array([0.0015, 0.0050, 0.0100]),
                9 / 10,
                98 / 100,
                10,
            ),
            np.array([0.21076116, 0.90368241, 1.90992892]),
            decimal=7,
        )


class TestExposureIndexValues(unittest.TestCase):
    """
    Define :func:`colour_hdri.exposure.dsc.exposure_index_values` definition
    unit tests methods.
    """

    def test_exposure_index_values(self):
        """
        Test :func:`colour_hdri.exposure.dsc.exposure_index_values`
        definition.
        """

        np.testing.assert_array_almost_equal(
            exposure_index_values(
                np.array([0.16439371, 0.08810904, 0.09310904])
            ),
            np.array([60.82957797, 113.49573211, 107.40095699]),
            decimal=7,
        )


class TestExposureValue100(unittest.TestCase):
    """
    Define :func:`colour_hdri.exposure.dsc.exposure_value_100` definition
    unit tests methods.
    """

    def test_exposure_value_100(self):
        """
        Test :func:`colour_hdri.exposure.dsc.exposure_value_100`
        definition.
        """

        np.testing.assert_array_almost_equal(
            exposure_value_100(
                np.array([8, 5.6, 2.8]),
                np.array([1 / 250, 1 / 500, 1 / 1000]),
                np.array([100, 800, 1600]),
            ),
            np.array([13.96578428, 10.93663794, 8.93663794]),
            decimal=7,
        )


class TestPhotometricExposureScaleFactorLagarde2014(unittest.TestCase):
    """
    Define :func:`colour_hdri.exposure.dsc.\
photometric_exposure_scale_factor_Lagarde2014` definition unit tests
    methods.
    """

    def test_photometric_exposure_scale_factor_Lagarde2014(self):
        """
        Test :func:`colour_hdri.exposure.dsc.\
photometric_exposure_scale_factor_Lagarde2014` definition.
        """

        np.testing.assert_array_almost_equal(
            photometric_exposure_scale_factor_Lagarde2014(
                np.array([13.96578428, 10.93663794, 8.93663794]),
                np.array([9 / 10, 8 / 10, 9 / 10]),
                np.array([98 / 100, 98 / 100, 94 / 100]),
                np.array([10, 10, 20]),
            ),
            np.array([0.00005221, 0.00037884, 0.00135554]),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
