# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour_hdri.utilities.exif` module."""

from __future__ import annotations

import numpy as np
import os
import shutil
import tempfile
import unittest

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.utilities import filter_files, vivified_to_dict
from colour_hdri.utilities import (
    EXIFTag,
    parse_exif_string,
    parse_exif_number,
    parse_exif_fraction,
    parse_exif_array,
    parse_exif_data,
    read_exif_tags,
    copy_exif_tags,
    update_exif_tags,
    delete_exif_tags,
    read_exif_tag,
    write_exif_tag,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES_FROBISHER_001",
    "TestParseExifString",
    "TestParseExifNumber",
    "TestParseExifFraction",
    "TestParseExifArray",
    "TestParseExifData",
    "TestReadExifTags",
    "TestCopyExifTags",
    "TestUpdateExifTags",
    "TestDeleteExifTags",
    "TestReadExifTag",
    "TestWriteExifTag",
]

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(
    ROOT_RESOURCES_TESTS, "frobisher_001"
)


class TestParseExifString(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.parse_exif_string` definition
    unit tests methods.
    """

    def test_parse_exif_string(self):
        """Test :func:`colour_hdri.utilities.exif.parse_exif_string` definition."""

        exif_tag = EXIFTag("EXIF", "Make", "Canon", "271")
        self.assertEqual(parse_exif_string(exif_tag), "Canon")


class TestParseExifNumber(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.parse_exif_number` definition
    unit tests methods.
    """

    def test_parse_exif_number(self):
        """Test :func:`colour_hdri.utilities.exif.parse_exif_number` definition."""

        exif_tag = EXIFTag("EXIF", "Focal Length", "16", "37386")
        self.assertEqual(parse_exif_number(exif_tag), 16)

        exif_tag = EXIFTag("EXIF", "Focal Length", "16", "37386")
        self.assertIsInstance(parse_exif_number(exif_tag, np.int_), np.int_)


class TestParseExifFraction(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.parse_exif_fraction` definition
    unit tests methods.
    """

    def test_parse_exif_fraction(self):
        """
        Test :func:`colour_hdri.utilities.exif.parse_exif_fraction`
        definition.
        """

        exif_tag = EXIFTag("EXIF", "Exposure Time", "0.01666666667", "33434")
        self.assertAlmostEqual(
            parse_exif_fraction(exif_tag), 0.01666666, places=7
        )

        exif_tag = EXIFTag("EXIF", "Exposure Time", "10/4000", "33434")
        self.assertAlmostEqual(
            parse_exif_fraction(exif_tag), 0.00250000, places=7
        )


class TestParseExifArray(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.parse_exif_array` definition
    unit tests methods.
    """

    def test_parse_exif_array(self):
        """Test :func:`colour_hdri.utilities.exif.parse_exif_array` definition."""

        exif_tag = EXIFTag(
            "EXIF",
            "Color Matrix 1",
            (
                "0.5309 -0.0229 -0.0336 "
                "-0.6241 1.3265 0.3337 "
                "-0.0817 0.1215 0.6664"
            ),
            "50721",
        )
        np.testing.assert_array_equal(
            parse_exif_array(exif_tag),
            np.array(
                [
                    0.5309,
                    -0.0229,
                    -0.0336,
                    -0.6241,
                    1.3265,
                    0.3337,
                    -0.0817,
                    0.1215,
                    0.6664,
                ]
            ),
        )

        np.testing.assert_array_equal(
            parse_exif_array(exif_tag, shape=(3, 3)),
            np.array(
                [
                    [0.5309, -0.0229, -0.0336],
                    [-0.6241, 1.3265, 0.3337],
                    [-0.0817, 0.1215, 0.6664],
                ]
            ),
        )


class TestParseExifData(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.parse_exif_data` definition
    unit tests methods.
    """

    def test_parse_exif_data(self):
        """Test :func:`colour_hdri.utilities.exif.parse_exif_data` definition."""

        self.assertListEqual(
            parse_exif_data(
                "[XMP]               - Description                     :"
            ),
            ["XMP", "-", "Description", ""],
        )

        self.assertListEqual(
            parse_exif_data(
                "[EXIF]            296 Resolution Unit                 : 2"
            ),
            ["EXIF", "296", "Resolution Unit", "2"],
        )

        self.assertListEqual(
            parse_exif_data(
                "[ICC_Profile]       8 Profile Version                 : 528"
            ),
            ["ICC_Profile", "8", "Profile Version", "528"],
        )


class TestReadExifTags(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.read_exif_tags` definition unit
    tests methods.
    """

    def test_read_exif_tags(self):
        """Test :func:`colour_hdri.utilities.exif.read_exif_tags` definition."""

        test_jpg_image = filter_files(ROOT_RESOURCES_FROBISHER_001, ("jpg",))[
            0
        ]
        exif_data = vivified_to_dict(read_exif_tags(test_jpg_image))

        self.assertIsInstance(exif_data, type(dict()))

        self.assertListEqual(
            sorted(exif_data.keys()),
            [
                "Composite",
                "EXIF",
                "ExifTool",
                "File",
                "ICC_Profile",
                "JFIF",
                "Photoshop",
                "XMP",
            ],
        )

        self.assertListEqual(
            sorted(exif_data["EXIF"].values(), key=lambda x: x[0].name),
            [
                [
                    EXIFTag(
                        "EXIF", "Camera Model Name", "EOS 5D Mark II", "272"
                    )
                ],
                [
                    EXIFTag(
                        "EXIF", "Create Date", "2015:09:19 03:39:20", "36868"
                    )
                ],
                [
                    EXIFTag(
                        "EXIF",
                        "Date/Time Original",
                        "2015:09:19 03:39:20",
                        "36867",
                    )
                ],
                [EXIFTag("EXIF", "Exif Image Height", "426", "40963")],
                [EXIFTag("EXIF", "Exif Image Width", "640", "40962")],
                [EXIFTag("EXIF", "Exposure Time", "0.125", "33434")],
                [EXIFTag("EXIF", "F Number", "8", "33437")],
                [EXIFTag("EXIF", "Focal Length", "16", "37386")],
                [EXIFTag("EXIF", "ISO", "100", "34855")],
                [EXIFTag("EXIF", "Make", "Canon", "271")],
                [EXIFTag("EXIF", "Modify Date", "2015:09:19 03:39:20", "306")],
                [EXIFTag("EXIF", "Orientation", "1", "274")],
                [EXIFTag("EXIF", "Photometric Interpretation", "2", "262")],
                [EXIFTag("EXIF", "Resolution Unit", "2", "296")],
                [EXIFTag("EXIF", "Software", "Photos 1.0.1", "305")],
                [EXIFTag("EXIF", "X Resolution", "72", "282")],
                [EXIFTag("EXIF", "Y Resolution", "72", "283")],
            ],
        )


class TestCopyExifTags(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.copy_exif_tags` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_copy_exif_tags(self):
        """Test :func:`colour_hdri.utilities.exif.copy_exif_tags` definition."""

        reference_jpg_image = filter_files(
            ROOT_RESOURCES_FROBISHER_001, ("jpg",)
        )[0]
        test_jpg_image = os.path.join(
            self._temporary_directory, os.path.basename(reference_jpg_image)
        )

        shutil.copyfile(reference_jpg_image, test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, "Aperture"), "8.0")
        delete_exif_tags(test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, "Aperture"), "")
        copy_exif_tags(reference_jpg_image, test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, "Aperture"), "8.0")


class TestUpdateExifTags(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.update_exif_tags` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_update_exif_tags(self):
        """Test :func:`colour_hdri.utilities.exif.update_exif_tags` definition."""

        reference_jpg_images = filter_files(
            ROOT_RESOURCES_FROBISHER_001, ("jpg",)
        )
        test_jpg_images = []
        for reference_jpg_image in reference_jpg_images:
            test_jpg_image = os.path.join(
                self._temporary_directory,
                os.path.basename(reference_jpg_image),
            )
            shutil.copyfile(reference_jpg_image, test_jpg_image)
            delete_exif_tags(test_jpg_image)
            self.assertEqual(read_exif_tag(test_jpg_image, "Aperture"), "")
            test_jpg_images.append(test_jpg_image)

        update_exif_tags(zip(reference_jpg_images, test_jpg_images))
        for test_jpg_image in test_jpg_images:
            self.assertEqual(read_exif_tag(test_jpg_image, "Aperture"), "8.0")


class TestDeleteExifTags(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.delete_exif_tags` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_delete_exif_tags(self):
        """Test :func:`colour_hdri.utilities.exif.delete_exif_tags` definition."""

        reference_jpg_image = filter_files(
            ROOT_RESOURCES_FROBISHER_001, ("jpg",)
        )[0]
        test_jpg_image = os.path.join(
            self._temporary_directory, os.path.basename(reference_jpg_image)
        )

        shutil.copyfile(reference_jpg_image, test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, "Aperture"), "8.0")
        delete_exif_tags(test_jpg_image)
        self.assertEqual(read_exif_tag(test_jpg_image, "Aperture"), "")


class TestReadExifTag(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.read_exif_tag` definition unit
    tests methods.
    """

    def test_read_exif_tag(self):
        """Test :func:`colour_hdri.utilities.exif.read_exif_tag` definition."""

        test_jpg_image = filter_files(ROOT_RESOURCES_FROBISHER_001, ("jpg",))[
            0
        ]

        self.assertEqual(read_exif_tag(test_jpg_image, "Aperture"), "8.0")
        self.assertEqual(read_exif_tag(test_jpg_image, "ExposureTime"), "1/8")
        self.assertEqual(read_exif_tag(test_jpg_image, "ISO"), "100")


class TestWriteExifTag(unittest.TestCase):
    """
    Define :func:`colour_hdri.utilities.exif.write_exif_tag` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_exif_tag(self):
        """Test :func:`colour_hdri.utilities.exif.write_exif_tag` definition."""

        reference_jpg_image = filter_files(
            ROOT_RESOURCES_FROBISHER_001, ("jpg",)
        )[0]
        test_jpg_image = os.path.join(
            self._temporary_directory, os.path.basename(reference_jpg_image)
        )

        shutil.copyfile(reference_jpg_image, test_jpg_image)
        # *Aperture* exif tag is not writeable, changing for *FNumber*.
        self.assertEqual(read_exif_tag(test_jpg_image, "FNumber"), "8.0")
        write_exif_tag(test_jpg_image, "FNumber", "16.0")
        self.assertEqual(read_exif_tag(test_jpg_image, "FNumber"), "16.0")


if __name__ == "__main__":
    unittest.main()
