"""Define the unit tests for the :mod:`colour_hdri.process.adobe_dng` module."""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
import zipfile

import numpy as np
from colour import read_image
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.hints import List

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.process import (
    convert_dng_files_to_intermediate_files,
    convert_raw_files_to_dng_files,
    read_dng_files_exif_tags,
)
from colour_hdri.utilities import filter_files

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES_FROBISHER_001",
    "ROOT_PROCESS",
    "IMAGES_RAW",
    "TestConvertRawFilesToDngFiles",
    "TestConvertDngFilesToIntermediateFiles",
]

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(ROOT_RESOURCES_TESTS, "frobisher_001")

ROOT_PROCESS: str = os.path.join(ROOT_RESOURCES_TESTS, "colour_hdri", "process")

IMAGES_RAW: List[str] = filter_files(ROOT_RESOURCES_FROBISHER_001, ("CR2",))


class TestConvertRawFilesToDngFiles:
    """
    Define :func:`colour_hdri.process.adobe_dng.\
convert_raw_files_to_dng_files` definition unit tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def teardown_method(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_convert_raw_files_to_dng_files(self):
        """
        Test :func:`colour_hdri.process.adobe_dng.\
convert_raw_files_to_dng_files` definition.
        """

        if platform.system() not in ("Windows", "Microsoft"):
            # *Adobe DNG Converter* is not available on *Linux*, and is not
            # reproducible on *macOS* thus we skip this unit test.
            return

        reference_dng_files = sorted(filter_files(ROOT_PROCESS, ("dng",)))
        test_dng_files = sorted(
            convert_raw_files_to_dng_files(IMAGES_RAW, self._temporary_directory)
        )

        for test_dng_file, reference_dng_file in zip(
            test_dng_files, reference_dng_files
        ):
            np.testing.assert_allclose(
                read_image(str(test_dng_file)),
                read_image(str(reference_dng_file)),
                atol=0.005,
            )


class TestConvertDngFilesToIntermediateFiles:
    """
    Define :func:`colour_hdri.process.adobe_dng.\
convert_dng_files_to_intermediate_files` definition unit tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def teardown_method(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_convert_dng_files_to_intermediate_files(self):
        """
        Test :func:`colour_hdri.process.adobe_dng.\
convert_dng_files_to_intermediate_files` definition.
        """

        reference_dng_files = sorted(filter_files(ROOT_PROCESS, ("dng",)))
        tests_dng_files = [
            os.path.join(
                self._temporary_directory, os.path.basename(reference_dng_file)
            )
            for reference_dng_file in reference_dng_files
        ]
        for reference_dng_file, tests_dng_file in zip(
            reference_dng_files, tests_dng_files
        ):
            shutil.copyfile(reference_dng_file, tests_dng_file)

        reference_zip_files = sorted(filter_files(ROOT_PROCESS, ("zip",)))

        for reference_zip_file in reference_zip_files:
            with zipfile.ZipFile(reference_zip_file) as zip_file:
                tiff_file_name = os.path.basename(reference_zip_file).replace(
                    ".zip", ""
                )
                with open(
                    os.path.join(self._temporary_directory, tiff_file_name),
                    "wb",
                ) as reference_tiff_file:
                    reference_tiff_file.write(zip_file.read(tiff_file_name))

        reference_tiff_files = sorted(
            filter_files(self._temporary_directory, ("tiff",))
        )

        # for reference_tiff_file in reference_tiff_files:
        #     os.chdir(os.path.dirname(reference_tiff_file))
        #     with zipfile.ZipFile(
        #             '{0}.zip'.format(reference_tiff_file),
        #             mode='w') as zip_file:
        #         zip_file.write(
        #             os.path.basename(reference_tiff_file),
        #             compress_type=zipfile.ZIP_DEFLATED)
        #         os.remove(reference_tiff_file)

        test_tiff_files = sorted(
            convert_dng_files_to_intermediate_files(
                tests_dng_files, self._temporary_directory
            )
        )

        for test_tiff_file, reference_tiff_file in zip(
            test_tiff_files, reference_tiff_files
        ):
            np.testing.assert_allclose(
                read_image(str(test_tiff_file)),
                read_image(str(reference_tiff_file)),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )


class TestReadDngFilesExifTags:
    """
    Define :func:`colour_hdri.process.adobe_dng.\
read_dng_files_exif_tags` definition unit tests methods.
    """

    def test_read_dng_files_exif_tags(self):
        """
        Test :func:`colour_hdri.process.adobe_dng.\
read_dng_files_exif_tags` definition.
        """

        reference_dng_files = sorted(filter_files(ROOT_PROCESS, ("dng",)))
        exif_tags = read_dng_files_exif_tags(reference_dng_files)
        assert len(exif_tags) == 3
        assert "EXIF" in exif_tags[0]
        assert "Make" in exif_tags[0]["EXIF"]

        np.testing.assert_allclose(
            exif_tags[0]["EXIF"]["Exposure Time"],
            0.12500000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_array_equal(
            exif_tags[0]["EXIF"]["Reduction Matrix 1"], np.identity(3)
        )
