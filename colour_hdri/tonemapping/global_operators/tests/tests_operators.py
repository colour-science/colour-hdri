# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for
:mod:`colour_hdri.tonemapping.global_operators.operators` module.
"""

from __future__ import division, unicode_literals

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
    tonemapping_operator_filmic)
from colour_hdri.tonemapping.global_operators.operators import log_average

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogAverage',
           'TestTonemappingOperatorSimple',
           'TestTonemappingOperatorNormalisation',
           'TestTonemappingOperatorGamma',
           'TestTonemappingOperatorLogarithmic',
           'TestTonemappingOperatorExponential',
           'TestTonemappingOperatorLogarithmicMapping',
           'TestTonemappingOperatorExponentiationMapping',
           'TestTonemappingOperatorSchlick1994',
           'TestTonemappingOperatorTumblin1999',
           'TestTonemappingOperatorReinhard2004',
           'TestTonemappingOperatorFilmic']


class TestLogAverage(unittest.TestCase):
    """
    Defines :func:`colour_hdri.utilities.common.log_average` definition unit
    tests methods.
    """

    def test_log_average(self):
        """
        Tests :func:`colour_hdri.utilities.common.log_average` definition.
        """

        np.testing.assert_almost_equal(
            log_average(np.linspace(0, 10, 10)),
            np.array(0.125071409675722),
            decimal=7)


class TestTonemappingOperatorSimple(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_simple` definition unit tests methods.
    """

    def test_tonemapping_operator_simple(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_simple` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_simple(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]])),
            np.array([[[0.32453826, 0.26011564, 0.19115324],
                       [0.58306189, 0.35678395, 0.28089936]],
                      [[0.81502903, 0.68316922, 0.57333401],
                       [0.86831276, 0.7746486, 0.68932119]]]),
            decimal=7)


class TestTonemappingOperatorNormalisation(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_normalisation` definition unit tests methods.
    """

    def test_tonemapping_operator_normalisation(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_normalisation` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_normalisation(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]])),
            np.array([[[0.1194995, 0.08743867, 0.05877821],
                       [0.34781155, 0.13795878, 0.09715429]],
                      [[1.09589878, 0.53629264, 0.3342109],
                       [1.63996061, 0.85495921, 0.55183713]]]),
            decimal=7)


class TestTonemappingOperatorGamma(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_gamma` definition unit tests methods.
    """

    def test_tonemapping_operator_gamma(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_gamma` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_gamma(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                0.5, -5),
            np.array([[[2.25439668e-04, 1.20699447e-04, 5.45419730e-05],
                       [1.90979251e-03, 3.00467090e-04, 1.49012329e-04]],
                      [[1.89600325e-02, 4.54048111e-03, 1.76335357e-03],
                       [4.24585372e-02, 1.15395529e-02, 4.80750815e-03]]]),
            decimal=7)


class TestTonemappingOperatorLogarithmic(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_logarithmic` definition unit tests methods.
    """

    def test_tonemapping_operator_logarithmic(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_logarithmic` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_logarithmic(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                10, 25),
            np.array([[[0.43458117, 0.31798628, 0.2137574],
                       [0.88293398, 0.35021407, 0.24663019]],
                      [[1.21686625, 0.59548969, 0.37110176],
                       [1.31992727, 0.68811651, 0.44414779]]]),
            decimal=7)


class TestTonemappingOperatorExponential(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_exponential` definition unit tests methods.
    """

    def test_tonemapping_operator_exponential(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_exponential` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_exponential(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                10, 25),
            np.array([[[0.14069744, 0.10294936, 0.06920484],
                       [0.38838746, 0.15405314, 0.10848838]],
                      [[0.94076798, 0.46037733, 0.28690142],
                       [1.16837291, 0.60910681, 0.39315063]]]),
            decimal=7)


class TestTonemappingOperatorLogarithmicMapping(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_logarithmic_mapping` definition unit tests methods.
    """

    def test_tonemapping_operator_logarithmic_mapping(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_logarithmic_mapping` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_logarithmic_mapping(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                10, 2.0),
            np.array([[[0.83661324, 0.61215614, 0.41150488],
                       [1.45740822, 0.57807818, 0.40709824]],
                      [[1.60813039, 0.78696, 0.49042367],
                       [1.63996061, 0.85495921, 0.55183713]]]),
            decimal=7)


class TestTonemappingOperatorExponentiationMapping(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_exponentiation_mapping` definition unit tests methods.
    """

    def test_tonemapping_operator_exponentiation_mapping(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_exponentiation_mapping` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_exponentiation_mapping(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                0.5, 2.0),
            np.array([[[0.71427077, 0.52263724, 0.3513283],
                       [1.2605181, 0.49998209, 0.35210087]],
                      [[1.53031135, 0.74887822, 0.46669158],
                       [1.63996061, 0.85495921, 0.55183713]]]),
            decimal=7)


class TestTonemappingOperatorSchlick1994(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Schlick1994` definition unit tests methods.
    """

    def test_tonemapping_operator_Schlick1994(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Schlick1994` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_Schlick1994(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                10),
            np.array([[[0.65311499, 0.4778891, 0.32124761],
                       [1.32918729, 0.52721959, 0.37128225]],
                      [[1.6196443, 0.79259448, 0.49393501],
                       [1.63996061, 0.85495921, 0.55183713]]]),
            decimal=7)


class TestTonemappingOperatorTumblin1999(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Tumblin1999` definition unit tests methods.
    """

    def test_tonemapping_operator_Tumblin1999(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Tumblin1999` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_Tumblin1999(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                50, 75, 85),
            np.array([[[0.11466371, 0.08390029, 0.05639963],
                       [0.28234863, 0.11199304, 0.07886852]],
                      [[0.64682852, 0.31653414, 0.19726013],
                       [0.8657459, 0.45133854, 0.29131842]]]),
            decimal=7)


class TestTonemappingOperatorReinhard2004(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Reinhard2004` definition unit tests methods.
    """

    def test_tonemapping_operator_Reinhard2004(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_Reinhard2004` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_Reinhard2004(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                -5, 0.5, 0.5, 0.5),
            np.array([[[0.03388145, 0.03028025, 0.02267653],
                       [0.08415536, 0.04335064, 0.03388967]],
                      [[0.14417912, 0.09625765, 0.07082831],
                       [0.15459508, 0.11021455, 0.08524267]]]),
            decimal=7)


class TestTonemappingOperatorFilmic(unittest.TestCase):
    """
    Defines :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_filmic` definition unit tests methods.
    """

    def test_tonemapping_operator_filmic(self):
        """
        Tests :func:`colour_hdri.tonemapping.global_operators.operators.\
tonemapping_operator_filmic` definition.
        """

        np.testing.assert_almost_equal(
            tonemapping_operator_filmic(
                np.array([[[0.48046875, 0.35156256, 0.23632812],
                           [1.39843753, 0.55468757, 0.39062594]],
                          [[4.40625388, 2.15625895, 1.34375372],
                           [6.59375023, 3.43751395, 2.21875829]]]),
                0.5, 0.5, 0.5, 0.5, 0.05, 0.5, 4, 15),
            np.array([[[0.77200408, 0.69565383, 0.58310785],
                       [0.93794714, 0.80274647, 0.72279964]],
                      [[1.00563196, 0.97235243, 0.93403169],
                       [1.01635759, 0.99657186, 0.97416808]]]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
