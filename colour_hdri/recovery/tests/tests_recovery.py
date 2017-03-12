# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour_hdri.recovery.highlights` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

from colour import read_image

from colour_hdri import TESTS_RESOURCES_DIRECTORY
from colour_hdri.process import RAW_CONVERTER, RAW_D_CONVERSION_ARGUMENTS
from colour_hdri.recovery import (
    highlights_recovery_blend,
    highlights_recovery_LCHab)
from colour_hdri.models import camera_space_to_sRGB
from colour_hdri.utilities import filter_files

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FROBISHER_001_DIRECTORY',
           'RECOVERY_DIRECTORY',
           'RAW_IMAGES',
           'XYZ_TO_CAMERA_SPACE_MATRIX',
           'TestHighlightsRecoveryBlend',
           'TestHighlightsRecoveryLCHab']

FROBISHER_001_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'frobisher_001')

RECOVERY_DIRECTORY = os.path.join(
    TESTS_RESOURCES_DIRECTORY, 'colour_hdri', 'recovery')

RAW_IMAGES = filter_files(FROBISHER_001_DIRECTORY, ('CR2',))

XYZ_TO_CAMERA_SPACE_MATRIX = np.array([
    [0.47160000, 0.06030000, -0.08300000],
    [-0.77980000, 1.54740000, 0.24800000],
    [-0.14960000, 0.19370000, 0.66510000]])


class TestHighlightsRecoveryBlend(unittest.TestCase):
    """
    Defines :func:`colour_hdri.recovery.highlights.\
highlights_recovery_blend` definition unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_highlights_recovery_blend(self):
        """
        Tests :func:`colour_hdri.recovery.highlights.highlights_recovery_blend`
        definition.
        """

        multipliers = np.array([2.42089718, 1.00000000, 1.54687415])
        multipliers /= np.max(multipliers)

        reference_raw_file = RAW_IMAGES[1]
        test_raw_file = os.path.join(
            self._temporary_directory, os.path.basename(reference_raw_file))
        shutil.copyfile(reference_raw_file, test_raw_file)
        command = [RAW_CONVERTER] + shlex.split(
            RAW_D_CONVERSION_ARGUMENTS.format(test_raw_file),
            posix=(False
                   if platform.system() in ('Windows', 'Microsoft') else
                   True))

        subprocess.call(command)

        test_tiff_file = read_image(
            str(re.sub('\.CR2$', '.tiff', test_raw_file)))[::10, ::10, :]

        test_tiff_file *= multipliers
        test_tiff_file = highlights_recovery_blend(
            test_tiff_file, multipliers)
        test_tiff_file = camera_space_to_sRGB(
            test_tiff_file, XYZ_TO_CAMERA_SPACE_MATRIX)

        reference_exr_file = read_image(str(os.path.join(
            RECOVERY_DIRECTORY,
            os.path.basename(re.sub('\.CR2$', '_Blend.exr', test_raw_file)))))

        np.testing.assert_allclose(
            test_tiff_file,
            reference_exr_file,
            rtol=0.0001,
            atol=0.0001)


class TestHighlightsRecoveryLCHab(unittest.TestCase):
    """
    Defines :func:`colour_hdri.recovery.highlights.\
highlights_recovery_LCHab` definition unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_highlights_recovery_LCHab(self):
        """
        Tests :func:`colour_hdri.recovery.highlights.highlights_recovery_LCHab`
        definition.
        """

        multipliers = np.array([2.42089718, 1.00000000, 1.54687415])
        multipliers /= np.max(multipliers)

        reference_raw_file = RAW_IMAGES[1]
        test_raw_file = os.path.join(
            self._temporary_directory, os.path.basename(reference_raw_file))
        shutil.copyfile(reference_raw_file, test_raw_file)
        command = [RAW_CONVERTER] + shlex.split(
            RAW_D_CONVERSION_ARGUMENTS.format(test_raw_file),
            posix=(False
                   if platform.system() in ('Windows', 'Microsoft') else
                   True))

        subprocess.call(command)

        test_tiff_file = read_image(
            str(re.sub('\.CR2$', '.tiff', test_raw_file)))[::10, ::10, :]

        test_tiff_file *= multipliers
        test_tiff_file = highlights_recovery_LCHab(
            test_tiff_file, min(multipliers))
        test_tiff_file = camera_space_to_sRGB(
            test_tiff_file, XYZ_TO_CAMERA_SPACE_MATRIX)

        reference_exr_file = read_image(str(os.path.join(
            RECOVERY_DIRECTORY,
            os.path.basename(re.sub('\.CR2$', '_LCHab.exr', test_raw_file)))))

        np.testing.assert_allclose(
            test_tiff_file,
            reference_exr_file,
            rtol=0.0001,
            atol=0.0001)


if __name__ == '__main__':
    unittest.main()
