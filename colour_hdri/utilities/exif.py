"""
EXIF Data Manipulation
======================

EXIF data manipulation routines based on *exiftool*:

-   :func:`colour_hdri.parse_exif_data`
-   :func:`colour_hdri.read_exif_tags`
-   :func:`colour_hdri.copy_exif_tags`
-   :func:`colour_hdri.update_exif_tags`
-   :func:`colour_hdri.delete_exif_tags`
-   :func:`colour_hdri.read_exif_tag`
-   :func:`colour_hdri.write_exif_tag`
"""

from __future__ import annotations

import logging
import numpy as np
import platform
import re
import subprocess  # nosec
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

from colour.hints import (
    Boolean,
    DTypeFloating,
    DTypeNumber,
    Floating,
    List,
    NDArray,
    Number,
    Optional,
    Sequence,
    SupportsIndex,
    Type,
    Union,
    cast,
)

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.utilities import as_array, as_float_scalar, optional
from colour.utilities.documentation import (
    DocstringText,
    is_documentation_building,
)

from colour_hdri.utilities import vivification

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "EXIF_EXECUTABLE",
    "EXIFTag",
    "parse_exif_string",
    "parse_exif_number",
    "parse_exif_fraction",
    "parse_exif_array",
    "parse_exif_data",
    "read_exif_tags",
    "copy_exif_tags",
    "update_exif_tags",
    "delete_exif_tags",
    "read_exif_tag",
    "write_exif_tag",
]

_IS_WINDOWS_PLATFORM: Boolean = platform.system() in ("Windows", "Microsoft")
"""Whether the current platform is *Windows*."""

EXIF_EXECUTABLE: str = "exiftool"
if is_documentation_building():  # pragma: no cover
    EXIF_EXECUTABLE = DocstringText(EXIF_EXECUTABLE)
    EXIF_EXECUTABLE.__doc__ = """
Command line EXIF manipulation application, usually Phil Harvey's *ExifTool*.
"""


@dataclass
class EXIFTag:
    """
    EXIF tag data.

    Parameters
    ----------
    group
        EXIF tag group name.
    name
        EXIF tag name.
    value
        EXIF tag value.
    identifier
        EXIF tag identifier.
    """

    group: Optional[str] = field(default_factory=lambda: None)
    name: Optional[str] = field(default_factory=lambda: None)
    value: Optional[str] = field(default_factory=lambda: None)
    identifier: Optional[str] = field(default_factory=lambda: None)


def parse_exif_string(exif_tag: EXIFTag) -> str:
    """
    Parse given EXIF tag assuming it is a string and return its value.

    Parameters
    ----------
    exif_tag
        EXIF tag to parse.

    Returns
    -------
    :class:`str`
        Parsed EXIF tag value.
    """

    return str(exif_tag.value)


def parse_exif_number(
    exif_tag: EXIFTag, dtype: Optional[Type[DTypeNumber]] = None
) -> Number:
    """
    Parse given EXIF tag assuming it is a number type and return its value.

    Parameters
    ----------
    exif_tag
        EXIF tag to parse.
    dtype
        Return value data type.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.integer`
        Parsed EXIF tag value.
    """

    dtype = cast(Type[DTypeNumber], optional(dtype, DEFAULT_FLOAT_DTYPE))

    return dtype(exif_tag.value)  # type: ignore[arg-type, return-value]


def parse_exif_fraction(
    exif_tag: EXIFTag, dtype: Optional[Type[DTypeFloating]] = None
) -> Floating:
    """
    Parse given EXIF tag assuming it is a fraction and return its value.

    Parameters
    ----------
    exif_tag
        EXIF tag to parse.
    dtype
        Return value data type.

    Returns
    -------
    :class:`numpy.floating`
        Parsed EXIF tag value.
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    value = (
        exif_tag.value
        if exif_tag.value is None
        else float(Fraction(exif_tag.value))
    )

    return as_float_scalar(value, dtype)  # type: ignore[arg-type]


def parse_exif_array(
    exif_tag: EXIFTag,
    dtype: Optional[Type[DTypeNumber]] = None,
    shape: Optional[Union[SupportsIndex, Sequence[SupportsIndex]]] = None,
) -> NDArray:
    """
    Parse given EXIF tag assuming it is an array and return its value.

    Parameters
    ----------
    exif_tag
        EXIF tag to parse.
    dtype
        Return value data type.
    shape
        Shape of the array to be returned.

    Returns
    -------
    :class:`numpy.ndarray`
        Parsed EXIF tag value.
    """

    dtype = cast(Type[DTypeNumber], optional(dtype, DEFAULT_FLOAT_DTYPE))

    value = (
        exif_tag.value if exif_tag.value is None else exif_tag.value.split()
    )

    return np.reshape(as_array(value, dtype), shape)  # type: ignore[arg-type]


def parse_exif_data(data: str) -> List:
    """
    Parse given EXIF data output from *exiftool*.

    Parameters
    ----------
    data
        EXIF data output.

    Returns
    -------
    :class:`list`
        Parsed EXIF data output.

    Raises
    ------
    ValueError
        If the EXIF data output cannot be parsed.
    """

    search = re.search(
        r"\[(?P<group>\w+)\]\s*(?P<id>(\d+|-))?(?P<tag>.*?):(?P<value>.*$)",
        data,
    )

    if search is not None:
        return [
            group.strip() if group is not None else group
            for group in (
                search.group("group"),
                search.group("id"),
                search.group("tag"),
                search.group("value"),
            )
        ]
    else:
        raise ValueError("The EXIF data output cannot be parsed!")


def read_exif_tags(image: str) -> defaultdict:
    """
    Return given image EXIF image tags.

    Parameters
    ----------
    image
        Image file.

    Returns
    -------
    :class:`defaultdict`
        EXIF tags.
    """

    logging.info(f"Reading '{image}' image EXIF data.")

    exif_tags = vivification()
    lines = str(
        subprocess.check_output(  # nosec
            [EXIF_EXECUTABLE, "-D", "-G", "-a", "-u", "-n", image],
            shell=_IS_WINDOWS_PLATFORM,
        ),
        "utf-8",
        "ignore",
    ).split("\n")

    for line in lines:
        if not line.strip():
            continue

        group, identifier, tag, value = parse_exif_data(line)

        if not exif_tags[group][tag]:
            exif_tags[group][tag] = []

        exif_tags[group][tag].append(EXIFTag(group, tag, value, identifier))

    return exif_tags


def copy_exif_tags(source: str, target: str) -> Boolean:
    """
    Copy given source image file EXIF tag to given image target.

    Parameters
    ----------
    source
        Source image file.
    target
        Target image file.

    Returns
    -------
    :class:`bool`
        Definition success.
    """

    logging.info(f"Copying '{source}' file EXIF data to '{target}' file.")

    arguments = [EXIF_EXECUTABLE, "-overwrite_original", "-TagsFromFile"]
    arguments += [source, target]
    subprocess.check_output(arguments, shell=_IS_WINDOWS_PLATFORM)  # nosec

    return True


# TODO: Find a better name.
def update_exif_tags(images: Sequence[Sequence[str]]) -> Boolean:
    """
    Update given images pairs EXIF tags.

    Parameters
    ----------
    images
        Image pairs to update the EXIF tags of.

    Returns
    -------
    :class:`bool`
        Definition success.
    """

    success = 1
    for (source, target) in images:
        success *= int(copy_exif_tags(source, target))

    return bool(success)


def delete_exif_tags(image: str) -> Boolean:
    """
    Delete all given image EXIF tags.

    Parameters
    ----------
    image
        Image file to delete the EXIF tags from.

    Returns
    -------
    :class:`bool`
        Definition success.
    """

    logging.info(f"Deleting '{image}' image EXIF tags.")

    subprocess.check_output(  # nosec
        [EXIF_EXECUTABLE, "-overwrite_original", "-all=", image],
        shell=_IS_WINDOWS_PLATFORM,
    )

    return True


def read_exif_tag(image: str, tag: str) -> str:
    """
    Return given image EXIF tag value.

    Parameters
    ----------
    image : str
        Image file to read the EXIF tag value of.
    tag : str
        Tag to read the value of.

    Returns
    -------
    :class:`str`
        Tag value.
    """

    value = (
        str(
            subprocess.check_output(  # nosec
                [EXIF_EXECUTABLE, f"-{tag}", image], shell=_IS_WINDOWS_PLATFORM
            ),
            "utf-8",
            "ignore",
        )
        .split(":")
        .pop()
        .strip()
    )

    logging.info(f"Reading '{image}' image '{tag}' EXIF tag value: '{value}'")

    return value


def write_exif_tag(image: str, tag: str, value: str) -> Boolean:
    """
    Set given image EXIF tag value.

    Parameters
    ----------
    image : str
        Image file to set the EXIF tag value of.
    tag : str
        Tag to set the value of.
    value : str
        Value to set.

    Returns
    -------
    :class:`bool`
        Definition success.
    """

    logging.info(
        f"Writing '{image}' image '{tag}' EXIF tag with '{value}' value."
    )

    arguments = [EXIF_EXECUTABLE, "-overwrite_original"]
    arguments += [f"-{tag}={value}", image]
    subprocess.check_output(arguments, shell=_IS_WINDOWS_PLATFORM)  # nosec

    return True
