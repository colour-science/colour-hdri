# !/usr/bin/env python
"""
Define the unit tests for the
:mod:`colour_hdri.sampling.variance_minimization` module.
"""

from __future__ import annotations

import numpy as np
import os
import unittest

from colour import RGB_COLOURSPACES, RGB_luminance, read_image

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.sampling import (
    light_probe_sampling_variance_minimization_Viriyothai2009,
)
from colour_hdri.sampling.variance_minimization import (
    luminance_variance,
    find_regions_variance_minimization_Viriyothai2009,
    highlight_regions_variance_minimization,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES_SAMPLING",
    "TestLuminanceVariance",
    "TestFindRegionsVarianceMinimizationViriyothai2009",
    "TestLightProbeSamplingVarianceMinimizationViriyothai2009",
]

ROOT_RESOURCES_SAMPLING: str = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_hdri", "sampling"
)


class TestLuminanceVariance(unittest.TestCase):
    """
    Define :func:`colour_hdri.sampling.variance_minimization.\
luminance_variance` definition unit tests methods.
    """

    def test_luminance_variance(self):
        """
        Test :func:`colour_hdri.sampling.variance_minimization.\
luminance_variance` definition.
        """

        a = np.tile(np.arange(5), (5, 1))

        self.assertAlmostEqual(luminance_variance(a), 12.24744871, places=7)


class TestFindRegionsVarianceMinimizationViriyothai2009(unittest.TestCase):
    """
    Define :func:`colour_hdri.sampling.variance_minimization.\
find_regions_variance_minimization_Viriyothai2009` definition unit tests
    methods.
    """

    def test_find_regions_variance_minimization_Viriyothai2009(self):
        """
        Test :func:`colour_hdri.sampling.variance_minimization.\
find_regions_variance_minimization_Viriyothai2009` definition.
        """

        colourspace = RGB_COLOURSPACES["sRGB"]

        image = read_image(
            str(
                os.path.join(
                    ROOT_RESOURCES_SAMPLING,
                    "tests_light_probe_sampling_variance_minimization_"
                    "Viriyothai2009.exr",
                )
            )
        )

        Y = RGB_luminance(image, colourspace.primaries, colourspace.whitepoint)

        regions = find_regions_variance_minimization_Viriyothai2009(Y, n=1)
        self.assertListEqual(regions, [(0, 256, 0, 156), (0, 256, 156, 256)])

        regions = find_regions_variance_minimization_Viriyothai2009(Y, n=2)
        self.assertListEqual(
            regions,
            [
                (0, 97, 0, 156),
                (97, 256, 0, 156),
                (0, 100, 156, 256),
                (100, 256, 156, 256),
            ],
        )

        regions = find_regions_variance_minimization_Viriyothai2009(Y, n=4)
        self.assertListEqual(
            regions,
            [
                (0, 39, 0, 91),
                (39, 97, 0, 91),
                (0, 39, 91, 156),
                (39, 97, 91, 156),
                (97, 159, 0, 92),
                (97, 159, 92, 156),
                (159, 256, 0, 93),
                (159, 256, 93, 156),
                (0, 42, 156, 216),
                (42, 100, 156, 216),
                (0, 44, 216, 256),
                (44, 100, 216, 256),
                (100, 163, 156, 215),
                (100, 163, 215, 256),
                (163, 256, 156, 216),
                (163, 256, 216, 256),
            ],
        )


class TestHighlightRegionsVarianceMinimization(unittest.TestCase):
    """
    Define :func:`colour_hdri.sampling.variance_minimization.\
highlight_regions_variance_minimization` definition unit tests methods.
    """

    def test_highlight_regions_variance_minimization(self):
        """
        Test :func:`colour_hdri.sampling.variance_minimization.\
highlight_regions_variance_minimization` definition.
        """

        colourspace = RGB_COLOURSPACES["sRGB"]

        image = read_image(
            str(
                os.path.join(
                    ROOT_RESOURCES_SAMPLING,
                    "tests_light_probe_sampling_variance_minimization_"
                    "Viriyothai2009.exr",
                )
            )
        )

        Y = RGB_luminance(image, colourspace.primaries, colourspace.whitepoint)
        regions = find_regions_variance_minimization_Viriyothai2009(Y, n=4)
        np.testing.assert_array_almost_equal(
            highlight_regions_variance_minimization(image, regions),
            read_image(
                str(
                    os.path.join(
                        ROOT_RESOURCES_SAMPLING,
                        "tests_highlight_regions_variance_minimization.exr",
                    )
                )
            ),
            decimal=7,
        )


class TestLightProbeSamplingVarianceMinimizationViriyothai2009(
    unittest.TestCase
):
    """
    Define :func:`colour_hdri.sampling.variance_minimization.\
light_probe_sampling_variance_minimization_Viriyothai2009` definition unit
    tests methods.
    """

    def test_light_probe_sampling_variance_minimization_Viriyothai2009(self):
        """
        Test :func:`colour_hdri.sampling.variance_minimization.\
light_probe_sampling_variance_minimization_Viriyothai2009` definition.
        """

        image = read_image(
            str(
                os.path.join(
                    ROOT_RESOURCES_SAMPLING,
                    "tests_light_probe_sampling_variance_minimization_"
                    "Viriyothai2009.exr",
                )
            )
        )

        lights = light_probe_sampling_variance_minimization_Viriyothai2009(
            image
        )
        uvs = np.array([light[0] for light in lights])
        colours = np.array([light[1] for light in lights])
        indexes = np.array([light[2] for light in lights])

        np.testing.assert_array_almost_equal(
            uvs,
            np.array(
                [
                    [0.16015625, 0.11328125],
                    [0.15625000, 0.32421875],
                    [0.42968750, 0.11328125],
                    [0.42968750, 0.32031250],
                    [0.16796875, 0.48437500],
                    [0.43359375, 0.48437500],
                    [0.18359375, 0.74218750],
                    [0.44140625, 0.75390625],
                    [0.64843750, 0.11718750],
                    [0.64843750, 0.35156250],
                    [0.87500000, 0.11718750],
                    [0.87500000, 0.35937500],
                    [0.64062500, 0.54687500],
                    [0.87500000, 0.54687500],
                    [0.64843750, 0.79687500],
                    [0.87500000, 0.80078125],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            colours,
            np.array(
                [
                    [40.65992016, 226.11660475, 266.78098774],
                    [30.73776919, 130.37145448, 161.10773420],
                    [98.10281688, 201.29676312, 299.40549910],
                    [74.89445847, 117.00525796, 191.89859456],
                    [42.24291545, 125.58142895, 167.82099468],
                    [90.15780473, 102.82409275, 192.97436064],
                    [82.17253280, 97.55175847, 179.72394174],
                    [152.10083479, 72.06310755, 224.16233468],
                    [128.44782221, 173.29928458, 301.74819988],
                    [105.67531514, 104.05130512, 209.73328489],
                    [196.73262107, 196.16986090, 392.89716852],
                    [154.23001331, 111.56532115, 265.79564852],
                    [120.04539376, 81.22123903, 201.26542896],
                    [191.57947493, 95.21154106, 286.79548484],
                    [168.29435712, 45.09299320, 213.38641733],
                    [253.65272349, 50.30476046, 303.96245855],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_equal(
            indexes,
            np.array(
                [
                    [29, 41],
                    [83, 40],
                    [29, 110],
                    [82, 110],
                    [124, 43],
                    [124, 111],
                    [190, 47],
                    [193, 113],
                    [30, 166],
                    [90, 166],
                    [30, 224],
                    [92, 224],
                    [140, 164],
                    [140, 224],
                    [204, 166],
                    [205, 224],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
