# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour_hdri.recovery.highlights` module."""

from __future__ import annotations

import numpy as np
import os
import platform
import re
import shlex
import shutil
import subprocess  # nosec
import tempfile
import unittest

from colour import read_image
from colour.hints import Boolean, List, NDArray

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.process import (
    RAW_CONVERTER,
    RAW_CONVERTER_ARGUMENTS_DEMOSAICING,
)
from colour_hdri.recovery import (
    highlights_recovery_blend,
    highlights_recovery_LCHab,
)
from colour_hdri.models import camera_space_to_sRGB
from colour_hdri.utilities import filter_files

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES_FROBISHER_001",
    "ROOT_RESOURCES_RECOVERY",
    "IMAGES_RAW",
    "matrix_XYZ_to_camera_space",
    "TestHighlightsRecoveryBlend",
    "TestHighlightsRecoveryLCHab",
]

_IS_WINDOWS_PLATFORM: Boolean = platform.system() in ("Windows", "Microsoft")
"""Whether the current platform is *Windows*."""

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(
    ROOT_RESOURCES_TESTS, "frobisher_001"
)

ROOT_RESOURCES_RECOVERY: str = os.path.join(
    ROOT_RESOURCES_TESTS, "colour_hdri", "recovery"
)

IMAGES_RAW: List[str] = filter_files(ROOT_RESOURCES_FROBISHER_001, ("CR2",))

matrix_XYZ_to_camera_space: NDArray = np.array(
    [
        [0.47160000, 0.06030000, -0.08300000],
        [-0.77980000, 1.54740000, 0.24800000],
        [-0.14960000, 0.19370000, 0.66510000],
    ]
)


class TestHighlightsRecoveryBlend(unittest.TestCase):
    """
    Define :func:`colour_hdri.recovery.highlights.\
highlights_recovery_blend` definition unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_highlights_recovery_blend(self):
        """
        Test :func:`colour_hdri.recovery.highlights.highlights_recovery_blend`
        definition.
        """

        multipliers = np.array([2.42089718, 1.00000000, 1.54687415])
        multipliers /= np.max(multipliers)

        reference_raw_file = IMAGES_RAW[1]
        test_raw_file = os.path.join(
            self._temporary_directory, os.path.basename(reference_raw_file)
        )
        shutil.copyfile(reference_raw_file, test_raw_file)
        command = [RAW_CONVERTER] + shlex.split(
            RAW_CONVERTER_ARGUMENTS_DEMOSAICING.format(raw_file=test_raw_file),
            posix=not _IS_WINDOWS_PLATFORM,
        )

        subprocess.call(command)  # nosec

        test_tiff_file = read_image(
            str(re.sub("\\.CR2$", ".tiff", test_raw_file))
        )[::10, ::10, :]

        test_tiff_file *= multipliers
        test_tiff_file = highlights_recovery_blend(test_tiff_file, multipliers)
        test_tiff_file = camera_space_to_sRGB(
            test_tiff_file, matrix_XYZ_to_camera_space
        )
        reference_exr_path = os.path.join(
            ROOT_RESOURCES_RECOVERY,
            os.path.basename(re.sub("\\.CR2$", "_Blend.exr", test_raw_file)),
        )
        reference_exr_file = read_image(str(reference_exr_path))

        np.testing.assert_allclose(
            test_tiff_file, reference_exr_file, rtol=0.0001, atol=0.0001
        )


class TestHighlightsRecoveryLCHab(unittest.TestCase):
    """
    Define :func:`colour_hdri.recovery.highlights.\
highlights_recovery_LCHab` definition unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_highlights_recovery_LCHab(self):
        """
        Test :func:`colour_hdri.recovery.highlights.highlights_recovery_LCHab`
        definition.
        """

        multipliers = np.array([2.42089718, 1.00000000, 1.54687415])
        multipliers /= np.max(multipliers)

        reference_raw_file = IMAGES_RAW[1]
        test_raw_file = os.path.join(
            self._temporary_directory, os.path.basename(reference_raw_file)
        )
        shutil.copyfile(reference_raw_file, test_raw_file)
        command = [RAW_CONVERTER] + shlex.split(
            RAW_CONVERTER_ARGUMENTS_DEMOSAICING.format(raw_file=test_raw_file),
            posix=not _IS_WINDOWS_PLATFORM,
        )

        subprocess.call(command)  # nosec

        test_tiff_file = read_image(
            str(re.sub("\\.CR2$", ".tiff", test_raw_file))
        )[::10, ::10, :]

        test_tiff_file *= multipliers
        test_tiff_file = highlights_recovery_LCHab(
            test_tiff_file, min(multipliers)
        )
        test_tiff_file = camera_space_to_sRGB(
            test_tiff_file, matrix_XYZ_to_camera_space
        )

        reference_exr_path = os.path.join(
            ROOT_RESOURCES_RECOVERY,
            os.path.basename(re.sub("\\.CR2$", "_LCHab.exr", test_raw_file)),
        )
        reference_exr_file = read_image(str(reference_exr_path))

        np.testing.assert_allclose(
            test_tiff_file, reference_exr_file, rtol=0.0001, atol=0.0001
        )


if __name__ == "__main__":
    unittest.main()
