"""Define the unit tests for the :mod:`colour_hdri.utilities.common` module."""

from __future__ import annotations

import os

from colour_hdri import ROOT_RESOURCES_TESTS
from colour_hdri.utilities import (
    filter_files,
    path_exists,
    vivification,
    vivified_to_dict,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES_FROBISHER_001",
    "TestVivification",
    "TestVivifiedToDict",
    "TestPathExists",
    "TestFilterFiles",
]

ROOT_RESOURCES_FROBISHER_001: str = os.path.join(ROOT_RESOURCES_TESTS, "frobisher_001")


class TestVivification:
    """
    Define :func:`colour_hdri.utilities.common.vivification` definition unit
    tests methods.
    """

    def test_vivification(self):
        """Test :func:`colour_hdri.utilities.common.vivification` definition."""

        vivified = vivification()
        vivified["my"]["attribute"] = 1
        assert "attribute" in vivified["my"]
        assert vivified["my"]["attribute"] == 1


class TestVivifiedToDict:
    """
    Define :func:`colour_hdri.utilities.common.vivified_to_dict` definition
    unit tests methods.
    """

    def test_vivified_to_dict(self):
        """Test :func:`colour_hdri.utilities.common.vivified_to_dict` definition."""

        vivified = vivification()
        vivified["my"]["attribute"] = 1
        vivified_as_dict = vivified_to_dict(vivified)
        assert isinstance({}, type(vivified_as_dict))
        assert "attribute" in vivified_as_dict["my"]
        assert vivified_as_dict["my"]["attribute"] == 1


class TestPathExists:
    """
    Define :func:`colour_hdri.utilities.common.path_exists` definition unit
    tests methods.
    """

    def test_path_exists(self):
        """Test :func:`colour_hdri.utilities.common.path_exists` definition."""

        assert path_exists(__file__)
        assert not path_exists("")


class TestFilterFiles:
    """
    Define :func:`colour_hdri.utilities.common.filter_files` definition unit
    tests methods.
    """

    def test_filter_files(self):
        """Test :func:`colour_hdri.utilities.common.filter_files` definition."""

        raw_files = filter_files(ROOT_RESOURCES_FROBISHER_001, ("CR2", "jpg"))
        assert sorted(map(os.path.basename, raw_files)) == [
            "IMG_2600.CR2",
            "IMG_2600.jpg",
            "IMG_2601.CR2",
            "IMG_2601.jpg",
            "IMG_2602.CR2",
            "IMG_2602.jpg",
        ]
