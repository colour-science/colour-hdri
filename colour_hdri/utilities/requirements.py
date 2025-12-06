"""
Requirements Utilities
======================

Define the requirements utilities objects.
"""

from __future__ import annotations

import colour.utilities

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_rawpy_installed",
    "is_lensfunpy_installed",
    "is_opencv_installed",
]


def is_rawpy_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *rawpy* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *rawpy* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *rawpy* is installed.

    Raises
    ------
    :class:`ImportError`
        If *rawpy* is not installed.
    """

    try:  # pragma: no cover
        import rawpy  # noqa: F401, PLC0415
    except ImportError as exception:  # pragma: no cover
        if raise_exception:
            error = f'"rawpy" related API features are not available: "{exception}".'

            raise ImportError(error) from exception

        return False
    else:
        return True


def is_lensfunpy_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *lensfunpy* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *lensfunpy* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *lensfunpy* is installed.

    Raises
    ------
    :class:`ImportError`
        If *lensfunpy* is not installed.
    """

    try:  # pragma: no cover
        import lensfunpy  # noqa: F401, PLC0415
    except ImportError as exception:  # pragma: no cover
        if raise_exception:
            error = (
                f'"lensfunpy" related API features are not available: "{exception}".'
            )

            raise ImportError(error) from exception

        return False
    else:
        return True


def is_opencv_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *OpenCV* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *OpenCV* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *OpenCV* is installed.

    Raises
    ------
    :class:`ImportError`
        If *OpenCV* is not installed.
    """

    try:  # pragma: no cover
        import cv2  # noqa: F401, PLC0415
    except ImportError as exception:  # pragma: no cover
        if raise_exception:
            error = f'"OpenCV" related API features are not available: "{exception}".'

            raise ImportError(error) from exception

        return False
    else:
        return True


colour.utilities.requirements.REQUIREMENTS_TO_CALLABLE.update(
    {
        "rawpy": is_rawpy_installed,
        "lensfunpy": is_lensfunpy_installed,
        "OpenCV": is_opencv_installed,
    }
)
