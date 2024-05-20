"""Define the unit tests for the :mod:`colour_hdri.exposure.common` module."""


import numpy as np
from colour.constants import TOLERANCE_ABSOLUTE_TESTS

from colour_hdri.exposure import (
    adjust_exposure,
    average_illuminance,
    average_luminance,
    illuminance_to_exposure_value,
    luminance_to_exposure_value,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestAverageLuminance",
    "TestAverageIlluminance",
    "TestLuminanceToExposureValue",
    "TestAdjustExposure",
]


class TestAverageLuminance:
    """
    Define :func:`colour_hdri.exposure.common.average_luminance` definition
    unit tests methods.
    """

    def test_average_luminance(self):
        """Test :func:`colour_hdri.exposure.common.average_luminance` definition."""

        np.testing.assert_allclose(
            average_luminance(
                np.array([2.8, 5.6, 8]),
                np.array([0.125, 0.5, 1.0]),
                np.array([100, 800, 16000]),
            ),
            np.array([7.84000000, 0.98000000, 0.05000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestAverageIlluminance:
    """
    Define :func:`colour_hdri.exposure.common.average_illuminance` definition
    unit tests methods.
    """

    def test_average_illuminance(self):
        """
        Test :func:`colour_hdri.exposure.common.average_illuminance`
        definition.
        """

        np.testing.assert_allclose(
            average_illuminance(
                np.array([2.8, 5.6, 8]),
                np.array([0.125, 0.5, 1.0]),
                np.array([100, 800, 16000]),
            ),
            np.array([156.80000000, 19.60000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestLuminanceToExposureValue:
    """
    Define :func:`colour_hdri.exposure.common.luminance_to_exposure_value`
    definition unit tests methods.
    """

    def test_luminance_to_exposure_value(self):
        """
        Test :func:`colour_hdri.exposure.common.luminance_to_exposure_value`
        definition.
        """

        np.testing.assert_allclose(
            luminance_to_exposure_value(
                np.array([0.125, 0.250, 0.125]),
                np.array([100, 100, 100]),
                np.array([12.5, 12.5, 14]),
            ),
            np.array([0.00000000, 1.00000000, -0.16349873]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestIlluminanceToExposureValue:
    """
    Define :func:`colour_hdri.exposure.common.illuminance_to_exposure_value`
    definition unit tests methods.
    """

    def test_illuminance_to_exposure_value(self):
        """
        Test :func:`colour_hdri.exposure.common.illuminance_to_exposure_value`
        definition.
        """

        np.testing.assert_allclose(
            illuminance_to_exposure_value(
                np.array([2.5, 5.0, 0.125]),
                np.array([100, 100, 100]),
                np.array([250, 250, 340]),
            ),
            np.array([0.00000000, 1.00000000, -4.76553475]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestAdjustExposure:
    """
    Define :func:`colour_hdri.exposure.common.adjust_exposure` definition
    unit tests methods.
    """

    def test_adjust_exposure(self):
        """Test :func:`colour_hdri.exposure.common.adjust_exposure` definition."""

        np.testing.assert_allclose(
            adjust_exposure(np.array([0.25, 0.5, 0.75, 1]), 1),
            np.array([0.5, 1.0, 1.5, 2.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
