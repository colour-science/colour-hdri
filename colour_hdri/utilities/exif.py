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
import platform
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from colour.constants import DTYPE_FLOAT_DEFAULT
from colour.hints import (
    DTypeFloat,
    DTypeReal,
    List,
    NDArray,
    Real,
    Sequence,
    SupportsIndex,
    Type,
)
from colour.utilities import as_array, as_float_scalar, optional
from colour.utilities.documentation import (
    DocstringText,
    is_documentation_building,
)

from colour_hdri.utilities import vivification

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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

LOGGER = logging.getLogger(__name__)

_IS_WINDOWS_PLATFORM: bool = platform.system() in ("Windows", "Microsoft")
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

    group: str | None = field(default_factory=lambda: None)
    name: str | None = field(default_factory=lambda: None)
    value: str | None = field(default_factory=lambda: None)
    identifier: str | None = field(default_factory=lambda: None)


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


def parse_exif_number(exif_tag: EXIFTag, dtype: Type[DTypeReal] | None = None) -> Real:
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

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    return dtype(exif_tag.value)  # pyright: ignore


def parse_exif_fraction(
    exif_tag: EXIFTag, dtype: Type[DTypeFloat] | None = None
) -> float:
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

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    value = (
        exif_tag.value if exif_tag.value is None else float(Fraction(exif_tag.value))
    )

    return as_float_scalar(value, dtype)  # pyright: ignore


def parse_exif_array(
    exif_tag: EXIFTag,
    dtype: Type[DTypeReal] | None = None,
    shape: SupportsIndex | Sequence[SupportsIndex] | None = None,
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

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    value = exif_tag.value if exif_tag.value is None else exif_tag.value.split()

    array = as_array(value, dtype)  # pyright: ignore

    if shape is not None:
        array = np.reshape(array, shape)

    return array


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

    LOGGER.info('Reading "%s" image EXIF data.', image)

    exif_tags = vivification()
    lines = str(
        subprocess.check_output(
            [EXIF_EXECUTABLE, "-D", "-G", "-a", "-u", "-n", image],
            shell=_IS_WINDOWS_PLATFORM,  # noqa: S603
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


def copy_exif_tags(source: str, target: str) -> bool:
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

    LOGGER.info('Copying "%s" file EXIF data to "%s" file.', source, target)

    arguments = [EXIF_EXECUTABLE, "-overwrite_original", "-TagsFromFile"]
    arguments += [source, target]
    subprocess.check_output(
        arguments,
        shell=_IS_WINDOWS_PLATFORM,  # noqa: S603
    )

    return True


# TODO: Find a better name.
def update_exif_tags(images: Sequence[Sequence[str]]) -> bool:
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
    for source, target in images:
        success *= int(copy_exif_tags(source, target))

    return bool(success)


def delete_exif_tags(image: str) -> bool:
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

    LOGGER.info('Deleting "%s" image EXIF tags.', image)

    subprocess.check_output(
        [EXIF_EXECUTABLE, "-overwrite_original", "-all=", image],
        shell=_IS_WINDOWS_PLATFORM,  # noqa: S603
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
            subprocess.check_output(
                [EXIF_EXECUTABLE, f"-{tag}", image],
                shell=_IS_WINDOWS_PLATFORM,  # noqa: S603
            ),
            "utf-8",
            "ignore",
        )
        .split(":")
        .pop()
        .strip()
    )

    LOGGER.info(
        'Reading "%s" image "%s" EXIF tag value: "%s"',
        image,
        tag,
        value,
    )

    return value


def write_exif_tag(image: str, tag: str, value: str) -> bool:
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

    LOGGER.info(
        'Writing "%s" image "%s" EXIF tag with "%s" value.',
        image,
        tag,
        value,
    )

    arguments = [EXIF_EXECUTABLE, "-overwrite_original"]
    arguments += [f"-{tag}={value}", image]
    subprocess.check_output(
        arguments,
        shell=_IS_WINDOWS_PLATFORM,  # noqa: S603
    )

    return True
