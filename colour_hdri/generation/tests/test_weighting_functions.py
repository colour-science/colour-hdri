"""
Define the unit tests for the
:mod:`colour_hdri.generation.weighting_functions` module.
"""

import numpy as np
from colour.constants import TOLERANCE_ABSOLUTE_TESTS

from colour_hdri.generation import (
    double_sigmoid_anchored_function,
    hat_function,
    normal_distribution_function,
    weighting_function_Debevec1997,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestNormalDistributionFunction",
    "TestHatFunction",
    "TestWeightingFunctionDebevec1997",
    "TestDoubleSigmoidAnchoredFunction",
]


class TestNormalDistributionFunction:
    """
    Define :func:`colour_hdri.generation.weighting_functions.\
normal_distribution_function` definition unit tests methods.
    """

    def test_normal_distribution_function(self):
        """
        Test :func:`colour_hdri.generation.weighting_functions.\
normal_distribution_function` definition.
        """

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestHatFunction:
    """
    Define :func:`colour_hdri.generation.weighting_functions.hat_function`
    definition unit tests methods.
    """

    def test_hat_function(self):
        """
        Test :func:`colour_hdri.generation.weighting_functions.hat_function`
        definition.
        """

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestWeightingFunctionDebevec1997:
    """
    Define :func:`colour_hdri.generation.weighting_functions.\
weighting_function_Debevec1997` definition unit tests methods.
    """

    def test_weighting_function_Debevec1997(self):
        """
        Test :func:`colour_hdri.generation.weighting_functions.\
weighting_function_Debevec1997` definition.
        """

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestDoubleSigmoidAnchoredFunction:
    """
    Define :func:`colour_hdri.generation.weighting_functions.\
double_sigmoid_anchored_function` definition unit tests methods.
    """

    def test_double_sigmoid_anchored_function(self):
        """
        Test :func:`colour_hdri.generation.weighting_functions.\
double_sigmoid_anchored_function` definition.
        """

        np.testing.assert_allclose(
            double_sigmoid_anchored_function(np.linspace(0, 1, 100)),
            np.array(
                [
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    0.00097561,
                    0.00922993,
                    0.02828386,
                    0.06125052,
                    0.11109001,
                    0.17981040,
                    0.26740859,
                    0.37091274,
                    0.48413098,
                    0.59858972,
                    0.70550702,
                    0.79791624,
                    0.87193997,
                    0.92683630,
                    0.96417075,
                    0.98673575,
                    0.99766181,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    0.99766181,
                    0.98673575,
                    0.96417075,
                    0.92683630,
                    0.87193997,
                    0.79791624,
                    0.70550702,
                    0.59858972,
                    0.48413098,
                    0.37091274,
                    0.26740859,
                    0.17981040,
                    0.11109001,
                    0.06125052,
                    0.02828386,
                    0.00922993,
                    0.00097561,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            double_sigmoid_anchored_function(np.linspace(0, 1, 100), 0, 0.1, 0.9, 1),
            np.array(
                [
                    0.00000000,
                    0.01246727,
                    0.06023189,
                    0.15898251,
                    0.31489864,
                    0.51009998,
                    0.70298770,
                    0.85350984,
                    0.94660553,
                    0.99009901,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    0.99009901,
                    0.94660553,
                    0.85350984,
                    0.70298770,
                    0.51009998,
                    0.31489864,
                    0.15898251,
                    0.06023189,
                    0.01246727,
                    0.00000000,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
