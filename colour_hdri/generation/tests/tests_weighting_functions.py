# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.generation.weighting_functions`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour_hdri.generation import (
    normal_distribution_function,
    hat_function,
    weighting_function_Debevec1997)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestNormalDistributionFunction',
           'TestHatFunction',
           'TestWeightingFunctionDebevec1997']


class TestNormalDistributionFunction(unittest.TestCase):
    """
    Defines :func:`colour_hdri.generation.weighting_functions.\
normal_distribution_function` definition unit tests methods.
    """

    def test_normal_distribution_function(self):
        """
        Tests :func:`colour_hdri.generation.weighting_functions.\
normal_distribution_function` definition.
        """

        np.testing.assert_almost_equal(
            normal_distribution_function(np.linspace(0, 1, 10)),
            np.array([0.0038659, 0.0347086, 0.1800217, 0.5394075, 0.9337121,
                      0.9337121, 0.5394075, 0.1800217, 0.0347086, 0.0038659]),
            decimal=7)

        np.testing.assert_almost_equal(
            normal_distribution_function(np.linspace(0, 1, 10), 0, 1),
            np.array([1.0000000, 0.9938462, 0.9756110, 0.9459595, 0.9059552,
                      0.8569969, 0.8007374, 0.7389913, 0.6736385, 0.6065307]),
            decimal=7)

        np.testing.assert_almost_equal(
            normal_distribution_function(np.linspace(0, 1, 10), 0.5, 0.5),
            np.array([0.6065307, 0.7389913, 0.8569969, 0.9459595, 0.9938462,
                      0.9938462, 0.9459595, 0.8569969, 0.7389913, 0.6065307]),
            decimal=7)


class TestHatFunction(unittest.TestCase):
    """
    Defines :func:`colour_hdri.generation.weighting_functions.hat_function`
    definition unit tests methods.
    """

    def test_hat_function(self):
        """
        Tests :func:`colour_hdri.generation.weighting_functions.hat_function`
        definition.
        """

        np.testing.assert_almost_equal(
            hat_function(np.linspace(0, 1, 10)),
            np.array([0., 0.9509921, 0.9991356, 0.9999981, 1.,
                      1., 0.9999981, 0.9991356, 0.9509921, 0.]),
            decimal=7)


class TestWeightingFunctionDebevec1997(unittest.TestCase):
    """
    Defines :func:`colour_hdri.generation.weighting_functions.\
weighting_function_Debevec1997` definition unit tests methods.
    """

    def test_weighting_function_Debevec1997(self):
        """
        Tests :func:`colour_hdri.generation.weighting_functions.\
weighting_function_Debevec1997` definition.
        """

        np.testing.assert_almost_equal(
            weighting_function_Debevec1997(np.linspace(0, 1, 10)),
            np.array([0., 0.2327366, 0.488491, 0.7442455, 1.,
                      1., 0.7442455, 0.488491, 0.2327366, 0.]),
            decimal=7)

        np.testing.assert_almost_equal(
            weighting_function_Debevec1997(np.linspace(0, 1, 10), 0, 1),
            np.array([0., 0.25, 0.5, 0.75, 1., 1., 0.75, 0.5, 0.25, 0.]),
            decimal=7)

        np.testing.assert_almost_equal(
            weighting_function_Debevec1997(np.linspace(0, 1, 10), 0.25, 0.75),
            np.array([0., 0., 0., 0.4285714, 1.,
                      1., 0.4285714, 0., 0., 0.]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
