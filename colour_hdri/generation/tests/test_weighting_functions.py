# !/usr/bin/env python
"""
Define the unit tests for the
:mod:`colour_hdri.generation.weighting_functions` module.
"""

import numpy as np
import unittest

from colour_hdri.generation import (
    normal_distribution_function,
    hat_function,
    weighting_function_Debevec1997,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestNormalDistributionFunction",
    "TestHatFunction",
    "TestWeightingFunctionDebevec1997",
]


class TestNormalDistributionFunction(unittest.TestCase):
    """
    Define :func:`colour_hdri.generation.weighting_functions.\
normal_distribution_function` definition unit tests methods.
    """

    def test_normal_distribution_function(self):
        """
        Test :func:`colour_hdri.generation.weighting_functions.\
normal_distribution_function` definition.
        """

        np.testing.assert_array_almost_equal(
            normal_distribution_function(np.linspace(0, 1, 10)),
            np.array(
                [
                    0.00386592,
                    0.03470859,
                    0.18002174,
                    0.53940751,
                    0.93371212,
                    0.93371212,
                    0.53940751,
                    0.18002174,
                    0.03470859,
                    0.00386592,
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            normal_distribution_function(np.linspace(0, 1, 10), 0, 1),
            np.array(
                [
                    1.00000000,
                    0.99384617,
                    0.97561098,
                    0.94595947,
                    0.90595519,
                    0.85699689,
                    0.80073740,
                    0.73899130,
                    0.67363846,
                    0.60653066,
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            normal_distribution_function(np.linspace(0, 1, 10), 0.5, 0.5),
            np.array(
                [
                    0.60653066,
                    0.73899130,
                    0.85699689,
                    0.94595947,
                    0.99384617,
                    0.99384617,
                    0.94595947,
                    0.85699689,
                    0.73899130,
                    0.60653066,
                ]
            ),
            decimal=7,
        )


class TestHatFunction(unittest.TestCase):
    """
    Define :func:`colour_hdri.generation.weighting_functions.hat_function`
    definition unit tests methods.
    """

    def test_hat_function(self):
        """
        Test :func:`colour_hdri.generation.weighting_functions.hat_function`
        definition.
        """

        np.testing.assert_array_almost_equal(
            hat_function(np.linspace(0, 1, 10)),
            np.array(
                [
                    0.00000000,
                    0.95099207,
                    0.99913557,
                    0.99999812,
                    1.00000000,
                    1.00000000,
                    0.99999812,
                    0.99913557,
                    0.95099207,
                    0.00000000,
                ]
            ),
            decimal=7,
        )


class TestWeightingFunctionDebevec1997(unittest.TestCase):
    """
    Define :func:`colour_hdri.generation.weighting_functions.\
weighting_function_Debevec1997` definition unit tests methods.
    """

    def test_weighting_function_Debevec1997(self):
        """
        Test :func:`colour_hdri.generation.weighting_functions.\
weighting_function_Debevec1997` definition.
        """

        np.testing.assert_array_almost_equal(
            weighting_function_Debevec1997(np.linspace(0, 1, 10)),
            np.array(
                [
                    0.00000000,
                    0.23273657,
                    0.48849105,
                    0.74424552,
                    1.00000000,
                    1.00000000,
                    0.74424552,
                    0.48849105,
                    0.23273657,
                    0.00000000,
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            weighting_function_Debevec1997(np.linspace(0, 1, 10), 0, 1),
            np.array(
                [
                    0.00000000,
                    0.25000000,
                    0.50000000,
                    0.75000000,
                    1.00000000,
                    1.00000000,
                    0.75000000,
                    0.50000000,
                    0.25000000,
                    0.00000000,
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            weighting_function_Debevec1997(np.linspace(0, 1, 10), 0.25, 0.75),
            np.array(
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.42857143,
                    1.00000000,
                    1.00000000,
                    0.42857143,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ]
            ),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
