# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.calibration.debevec1997` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import unittest

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.calibration import (
    g_solve,
    camera_response_functions_Debevec1997)
from colour_hdri.sampling import samples_Grossberg2003
from colour_hdri.utilities import ImageStack, average_luminance, filter_files

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FROBISHER_001_DIRECTORY',
           'CALIBRATION_DIRECTORY',
           'JPG_IMAGES',
           'TestGSolve',
           'TestCameraResponseFunctionsDebevec1997']

FROBISHER_001_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'frobisher_001')

CALIBRATION_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'colour_hdri', 'calibration')

JPG_IMAGES = filter_files(FROBISHER_001_DIRECTORY, ('jpg',))


class TestGSolve(unittest.TestCase):
    """
    Defines :func:`colour_hdri.calibration.debevec1997.g_solve` definition
    unit tests methods.
    """

    def test_g_solve(self):
        """
        Tests :func:`colour_hdri.calibration.debevec1997.g_solve` definition.
        """

        image_stack = ImageStack.from_files(JPG_IMAGES)
        L_l = np.log(average_luminance(image_stack.f_number,
                                       image_stack.exposure_time,
                                       image_stack.iso))
        samples = samples_Grossberg2003(image_stack.data)

        for i in range(3):
            g, lE = g_solve(samples[..., i], L_l)

            # Lower precision for unit tests under *travis-ci*.
            np.testing.assert_allclose(
                g[0:-2],
                np.load(os.path.join(
                    CALIBRATION_DIRECTORY,
                    'test_g_solve_g_{0}.npy'.format(i)))[0:-2],
                rtol=0.001,
                atol=0.001)

            # Lower precision for unit tests under *travis-ci*.
            np.testing.assert_allclose(
                lE[1:],
                np.load(os.path.join(
                    CALIBRATION_DIRECTORY,
                    'test_g_solve_lE_{0}.npy'.format(i)))[1:],
                rtol=0.001,
                atol=0.001)


class TestCameraResponseFunctionsDebevec1997(unittest.TestCase):
    """
    Defines :func:`colour_hdri.calibration.debevec1997.\
camera_response_functions_Debevec1997` definition unit tests methods.
    """

    def test_camera_response_function_Debevec1997(self):
        """
        Tests :func:`colour_hdri.calibration.debevec1997.\
camera_response_functions_Debevec1997` definition.
        """

        # Lower precision for unit tests under *travis-ci*.
        np.testing.assert_allclose(
            camera_response_functions_Debevec1997(
                ImageStack.from_files(JPG_IMAGES)),
            np.load(os.path.join(
                CALIBRATION_DIRECTORY,
                'test_camera_response_function_Debevec1997_crfs.npy')),
            rtol=0.00001,
            atol=0.00001)


if __name__ == '__main__':
    unittest.main()
