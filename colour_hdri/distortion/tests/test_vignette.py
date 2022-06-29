# !/usr/bin/env python
"""
Define the unit tests for the :mod:`colour_hdri.distortion.vignette` module.
"""

from __future__ import annotations

import numpy as np
import unittest

from colour_hdri.distortion import (
    apply_radial_gradient,
    parabolic_2D_function,
    hyperbolic_cosine_2D_function,
    characterise_vignette,
    correct_vignette,
)
from colour_hdri.distortion.vignette import vignette_center

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestApplyRadialGradient",
    "TestParabolic2DFunction",
    "TestHyperbolicCosine2DFunction",
    "TestVignetteCenter",
]


class TestApplyRadialGradient(unittest.TestCase):
    """
    Define :func:`colour_hdri.distortion.vignette.apply_radial_gradient`
    definition unit tests methods.
    """

    def test_apply_radial_gradient(self):
        """
        Test :func:`colour_hdri.distortion.vignette.apply_radial_gradient`
        definition.
        """

        np.testing.assert_array_almost_equal(
            apply_radial_gradient(np.ones([5, 7])),
            np.array(
                [
                    [
                        0.00000000,
                        0.02293883,
                        0.21176451,
                        0.28571429,
                        0.21176451,
                        0.02293883,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.24369618,
                        0.51146942,
                        0.64285714,
                        0.51146942,
                        0.24369618,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.33333333,
                        0.66666667,
                        1.0,
                        0.66666667,
                        0.33333333,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.24369618,
                        0.51146942,
                        0.64285714,
                        0.51146942,
                        0.24369618,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.02293883,
                        0.21176451,
                        0.28571429,
                        0.21176451,
                        0.02293883,
                        0.00000000,
                    ],
                ]
            ),
            decimal=7,
        )


class TestParabolic2DFunction(unittest.TestCase):
    """
    Define :func:`colour_hdri.distortion.vignette.parabolic_2D_function`
    definition unit tests methods.
    """

    def test_parabolic_2D_function(self):
        """
        Test :func:`colour_hdri.distortion.vignette.parabolic_2D_function`
        definition.
        """

        x_1, y_1 = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 7))
        np.testing.assert_array_almost_equal(
            parabolic_2D_function((x_1, y_1), -0.5, 0, 1, -0.5, 0, 1),
            np.array(
                [
                    [
                        1.00000000,
                        0.98437500,
                        0.93750000,
                        0.85937500,
                        0.75000000,
                    ],
                    [
                        0.99305556,
                        0.97743056,
                        0.93055556,
                        0.85243056,
                        0.74305556,
                    ],
                    [
                        0.97222222,
                        0.95659722,
                        0.90972222,
                        0.83159722,
                        0.72222222,
                    ],
                    [
                        0.93750000,
                        0.92187500,
                        0.87500000,
                        0.79687500,
                        0.68750000,
                    ],
                    [
                        0.88888889,
                        0.87326389,
                        0.82638889,
                        0.74826389,
                        0.63888889,
                    ],
                    [
                        0.82638889,
                        0.81076389,
                        0.76388889,
                        0.68576389,
                        0.57638889,
                    ],
                    [
                        0.75000000,
                        0.73437500,
                        0.68750000,
                        0.60937500,
                        0.50000000,
                    ],
                ]
            ),
            decimal=7,
        )


class TestHyperbolicCosine2DFunction(unittest.TestCase):
    """
    Define :func:`colour_hdri.distortion.vignette.hyperbolic_cosine_2D_function`
    definition unit tests methods.
    """

    def test_hyperbolic_cosine_2D_function(self):
        """
        Test :func:`colour_hdri.distortion.vignette.hyperbolic_cosine_2D_function`
        definition.
        """

        x_1, y_1 = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 7))
        np.testing.assert_array_almost_equal(
            hyperbolic_cosine_2D_function((x_1, y_1), 1, -0.5, 1, -0.5, 1),
            np.array(
                [
                    [
                        1.00000000,
                        0.96858690,
                        0.87237403,
                        0.70531672,
                        0.45691937,
                    ],
                    [
                        0.98607893,
                        0.95422853,
                        0.85667628,
                        0.68729334,
                        0.43543803,
                    ],
                    [
                        0.94392813,
                        0.91075364,
                        0.80914594,
                        0.63272141,
                        0.37039595,
                    ],
                    [
                        0.87237403,
                        0.83695181,
                        0.72845968,
                        0.54008151,
                        0.25998221,
                    ],
                    [
                        0.76942442,
                        0.73076823,
                        0.61237102,
                        0.40679437,
                        0.10112265,
                    ],
                    [
                        0.63221295,
                        0.58924652,
                        0.45764781,
                        0.22914897,
                        -0.11060571,
                    ],
                    [
                        0.45691937,
                        0.40844642,
                        0.25998221,
                        0.00219930,
                        -0.38109785,
                    ],
                ]
            ),
            decimal=7,
        )


class TestVignetteCenter(unittest.TestCase):
    """
    Define :func:`colour_hdri.distortion.vignette.vignette_center` definition
    unit tests methods.
    """

    def test_vignette_center(self):
        """
        Test :func:`colour_hdri.distortion.vignette.vignette_center` definition.
        """

        np.testing.assert_array_almost_equal(
            vignette_center(apply_radial_gradient(np.ones([50, 70, 3]))),
            np.array([0.49000000, 0.49285714]),
            decimal=7,
        )


class TestCharacteriseVignette(unittest.TestCase):
    """
    Define :func:`colour_hdri.distortion.vignette.characterise_vignette`
    definition unit tests methods.
    """

    def test_characterise_vignette(self):
        """
        Test :func:`colour_hdri.distortion.vignette.characterise_vignette`
        definition.
        """

        coefficients, center = characterise_vignette(
            apply_radial_gradient(np.ones([50, 70, 3]))
        ).values
        np.testing.assert_array_almost_equal(
            coefficients,
            np.array(
                [
                    [
                        -5.00000000,
                        0.06898022,
                        0.90000000,
                        -5.00000000,
                        0.04952344,
                        0.90000000,
                    ],
                    [
                        -5.00000000,
                        0.06898022,
                        0.90000000,
                        -5.00000000,
                        0.04952344,
                        0.90000000,
                    ],
                    [
                        -5.00000000,
                        0.06898022,
                        0.90000000,
                        -5.00000000,
                        0.04952344,
                        0.90000000,
                    ],
                ]
            ),
            decimal=7,
        )
        np.testing.assert_array_almost_equal(
            center,
            np.array([0.49000000, 0.49285714]),
            decimal=7,
        )

        coefficients, center = characterise_vignette(
            apply_radial_gradient(np.ones([50, 70, 3])),
            method="Hyperbolic Cosine",
        ).values
        np.testing.assert_array_almost_equal(
            coefficients,
            np.array(
                [
                    [
                        2.09353284,
                        -0.49000000,
                        1.47487009,
                        -0.49285714,
                        0.65106212,
                    ],
                    [
                        2.09353284,
                        -0.49000000,
                        1.47487009,
                        -0.49285714,
                        0.65106212,
                    ],
                    [
                        2.09353284,
                        -0.49000000,
                        1.47487009,
                        -0.49285714,
                        0.65106212,
                    ],
                ]
            ),
            decimal=7,
        )
        np.testing.assert_array_almost_equal(
            center,
            np.array([0.49000000, 0.49285714]),
            decimal=7,
        )


class TestCorrectVignette(unittest.TestCase):
    """
    Define :func:`colour_hdri.distortion.vignette.correct_vignette`
    definition unit tests methods.
    """

    def test_correct_vignette(self):
        """
        Test :func:`colour_hdri.distortion.vignette.correct_vignette`
        definition.
        """

        image = apply_radial_gradient(np.ones([5, 7]))
        characterisation_data = characterise_vignette(image)
        np.testing.assert_array_almost_equal(
            correct_vignette(image, characterisation_data),
            np.array(
                [
                    [
                        -0.00000000,
                        0.12207694,
                        0.59724084,
                        0.74726015,
                        0.78073305,
                        1.08008213,
                        -0.00000000,
                    ],
                    [
                        0.00000000,
                        0.41329379,
                        0.67626899,
                        0.81987841,
                        0.76000993,
                        0.57614485,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.46815842,
                        0.75871697,
                        1.10319977,
                        0.83821275,
                        0.61123607,
                        0.00000000,
                    ],
                    [
                        0.00000000,
                        0.43909067,
                        0.70873168,
                        0.85777601,
                        0.80125512,
                        0.62754066,
                        -0.00000000,
                    ],
                    [
                        -0.00000000,
                        0.19337964,
                        0.74228485,
                        0.91263519,
                        1.04857765,
                        -0.47743533,
                        -0.00000000,
                    ],
                ]
            ),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
