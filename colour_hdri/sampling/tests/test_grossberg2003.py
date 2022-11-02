# !/usr/bin/env python
"""
Define the unit tests for the :mod:`colour_hdri.sampling.grossberg2003`
module.
"""

from __future__ import annotations

import numpy as np
import os
import unittest

from colour.hints import List

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.sampling import samples_Grossberg2003
from colour_hdri.utilities import ImageStack, filter_files

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES_FROBISHER_001",
    "ROOT_RESOURCES_SAMPLING",
    "IMAGES_JPG",
    "TestSamplesGrossberg2003",
]

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(
    ROOT_RESOURCES_TESTS, "frobisher_001"
)

ROOT_RESOURCES_SAMPLING: str = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_hdri", "sampling"
)

IMAGES_JPG: List[str] = filter_files(ROOT_RESOURCES_FROBISHER_001, ("jpg",))


class TestSamplesGrossberg2003(unittest.TestCase):
    """
    Define :func:`colour_hdri.sampling.grossberg2003.\
samples_Grossberg2003` definition unit tests methods.
    """

    def test_samples_Grossberg2003(self):
        """
        Test :func:`colour_hdri.sampling.grossberg2003.\
samples_Grossberg2003` definition.
        """

        np.testing.assert_array_almost_equal(
            samples_Grossberg2003(ImageStack.from_files(IMAGES_JPG).data),
            np.load(
                os.path.join(
                    ROOT_RESOURCES_SAMPLING, "test_samples_Grossberg2003.npy"
                )
            ),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
