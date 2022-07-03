# !/usr/bin/env python
"""
Define the unit tests for the
:mod:`colour_hdri.tonemapping.global_operators.operators` module.
"""

import numpy as np
import unittest

from colour_hdri.tonemapping import (
    tonemapping_operator_simple,
    tonemapping_operator_normalisation,
    tonemapping_operator_gamma,
    tonemapping_operator_logarithmic,
    tonemapping_operator_exponential,
    tonemapping_operator_logarithmic_mapping,
    tonemapping_operator_exponentiation_mapping,
    tonemapping_operator_Schlick1994,
    tonemapping_operator_Tumblin1999,
    tonemapping_operator_Reinhard2004,
    tonemapping_operator_filmic,
)
from colour_hdri.tonemapping.global_operators.operators import log_average

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogAverage",
    "TestTonemappingOperatorSimple",
    "TestTonemappingOperatorNormalisation",
    "TestTonemappingOperatorGamma",
    "TestTonemappingOperatorLogarithmic",
    "TestTonemappingOperatorExponential",
    "TestTonemappingOperatorLogarithmicMapping",
    "TestTonemappingOperatorExponentiationMapping",
    "TestTonemappingOperatorSchlick1994",
    "TestTonemappingOperatorTumblin1999",
    "TestTonemappingOperatorReinhard2004",
    "TestTonemappingOperatorFilmic",
]


class TestLogAverage(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.common.log_average` definition unit
    tests methods.
    """

    def test_log_average(self):
        """Test :func:`colour_hdri.utilities.common.log_average` definition."""

        np.testing.assert_allclose(
            log_average(np.linspace(0, 10, 10)),
            np.array(0.125071409675722),
            rtol=0.025,
            atol=0.025,
        )


class TestTonemappingOperatorSimple(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_simple` definition unit tests methods.
    """

    def test_tonemapping_operator_simple(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_simple` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_simple(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                )
            ),
            np.array(
                [
                    [
                        [0.32453826, 0.26011564, 0.19115324],
                        [0.58306189, 0.35678395, 0.28089936],
                    ],
                    [
                        [0.81502903, 0.68316922, 0.57333401],
                        [0.86831276, 0.7746486, 0.68932119],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorNormalisation(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_normalisation` definition unit tests methods.
    """

    def test_tonemapping_operator_normalisation(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_normalisation` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_normalisation(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                )
            ),
            np.array(
                [
                    [
                        [0.11949973, 0.08743884, 0.05877832],
                        [0.34781223, 0.13795905, 0.09715449],
                    ],
                    [
                        [1.09590092, 0.53629369, 0.33421155],
                        [1.63996382, 0.85496088, 0.55183821],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorGamma(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_gamma` definition unit tests methods.
    """

    def test_tonemapping_operator_gamma(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_gamma` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_gamma(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                0.5,
                -5,
            ),
            np.array(
                [
                    [
                        [2.25439668e-04, 1.20699447e-04, 5.45419730e-05],
                        [1.90979251e-03, 3.00467090e-04, 1.49012329e-04],
                    ],
                    [
                        [1.89600325e-02, 4.54048111e-03, 1.76335357e-03],
                        [4.24585372e-02, 1.15395529e-02, 4.80750815e-03],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorLogarithmic(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_logarithmic` definition unit tests methods.
    """

    def test_tonemapping_operator_logarithmic(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_logarithmic` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_logarithmic(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                10,
                25,
            ),
            np.array(
                [
                    [
                        [0.43458200, 0.31798688, 0.21375781],
                        [0.88293445, 0.35021426, 0.24663032],
                    ],
                    [
                        [1.21686841, 0.59549075, 0.37110242],
                        [1.31992973, 0.68811779, 0.44414861],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorExponential(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_exponential` definition unit tests methods.
    """

    def test_tonemapping_operator_exponential(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_exponential` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_exponential(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                10,
                25,
            ),
            np.array(
                [
                    [
                        [0.14069770, 0.10294955, 0.06920496],
                        [0.38838808, 0.15405339, 0.10848855],
                    ],
                    [
                        [0.94076970, 0.46037817, 0.28690194],
                        [1.16837509, 0.60910795, 0.39315137],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorLogarithmicMapping(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_logarithmic_mapping` definition unit tests methods.
    """

    def test_tonemapping_operator_logarithmic_mapping(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_logarithmic_mapping` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_logarithmic_mapping(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                10,
                2.0,
            ),
            np.array(
                [
                    [
                        [0.83661533, 0.61215767, 0.41150591],
                        [1.45740883, 0.57807842, 0.40709841],
                    ],
                    [
                        [1.60813343, 0.78696148, 0.49042459],
                        [1.63996382, 0.85496088, 0.55183821],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorExponentiationMapping(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_exponentiation_mapping` definition unit tests methods.
    """

    def test_tonemapping_operator_exponentiation_mapping(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_exponentiation_mapping` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_exponentiation_mapping(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                0.5,
                2.0,
            ),
            np.array(
                [
                    [
                        [0.71427273, 0.52263867, 0.35132926],
                        [1.26051891, 0.49998241, 0.35210109],
                    ],
                    [
                        [1.5303143, 0.74887966, 0.46669247],
                        [1.63996382, 0.85496088, 0.55183821],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorSchlick1994(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Schlick1994` definition unit tests methods.
    """

    def test_tonemapping_operator_Schlick1994(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Schlick1994` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_Schlick1994(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                10,
            ),
            np.array(
                [
                    [
                        [0.65311658, 0.47789026, 0.32124839],
                        [1.32918845, 0.52722005, 0.37128257],
                    ],
                    [
                        [1.61964741, 0.79259601, 0.49393596],
                        [1.63996382, 0.85496088, 0.55183821],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorTumblin1999(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Tumblin1999` definition unit tests methods.
    """

    def test_tonemapping_operator_Tumblin1999(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Tumblin1999` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_Tumblin1999(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                50,
                75,
                85,
            ),
            np.array(
                [
                    [
                        [0.11466395, 0.08390046, 0.05639975],
                        [0.28234899, 0.11199318, 0.07886862],
                    ],
                    [
                        [0.64682951, 0.31653462, 0.19726043],
                        [0.86574718, 0.45133921, 0.29131885],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorReinhard2004(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Reinhard2004` definition unit tests methods.
    """

    def test_tonemapping_operator_Reinhard2004(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Reinhard2004` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_Reinhard2004(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                -5,
                0.5,
                0.5,
                0.5,
            ),
            np.array(
                [
                    [
                        [0.03388145, 0.03028025, 0.02267653],
                        [0.08415536, 0.04335064, 0.03388967],
                    ],
                    [
                        [0.14417912, 0.09625765, 0.07082831],
                        [0.15459508, 0.11021455, 0.08524267],
                    ],
                ]
            ),
            decimal=7,
        )


class TestTonemappingOperatorFilmic(unittest.TestCase):
    """
    Define :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_filmic` definition unit tests methods.
    """

    def test_tonemapping_operator_filmic(self):
        """
        Test :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_filmic` definition.
        """

        np.testing.assert_array_almost_equal(
            tonemapping_operator_filmic(
                np.array(
                    [
                        [
                            [0.48046875, 0.35156256, 0.23632812],
                            [1.39843753, 0.55468757, 0.39062594],
                        ],
                        [
                            [4.40625388, 2.15625895, 1.34375372],
                            [6.59375023, 3.43751395, 2.21875829],
                        ],
                    ]
                ),
                0.5,
                0.5,
                0.5,
                0.5,
                0.05,
                0.5,
                4,
                15,
            ),
            np.array(
                [
                    [
                        [0.77200408, 0.69565383, 0.58310785],
                        [0.93794714, 0.80274647, 0.72279964],
                    ],
                    [
                        [1.00563196, 0.97235243, 0.93403169],
                        [1.01635759, 0.99657186, 0.97416808],
                    ],
                ]
            ),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
