"""
Image Data & Metadata Utilities
===============================

Defines various image data and metadata utilities classes:

-   :class:`colour_hdri.Metadata`
-   :class:`colour_hdri.Image`
-   :class:`colour_hdri.ImageStack`
"""

from __future__ import annotations

import logging
import numpy as np
from collections.abc import MutableSequence
from dataclasses import dataclass, field, fields

from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    FloatingOrNDArray,
    Integer,
    List,
    NDArray,
    Optional,
    Sequence,
    Union,
    cast,
)
from colour import read_image
from colour.utilities import (
    MixinDataclassArray,
    as_float_array,
    is_string,
    tsplit,
    tstack,
    warning,
)

from colour_hdri.exposure import average_luminance
from colour_hdri.utilities.exif import (
    parse_exif_array,
    parse_exif_fraction,
    parse_exif_number,
    read_exif_tags,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Metadata",
    "Image",
    "ImageStack",
]


@dataclass
class Metadata(MixinDataclassArray):
    """
    Define the base object for storing exif metadata relevant to
    HDRI Generation.

    Parameters
    ----------
    f_number
        Image *FNumber*.
    exposure_time
        Image *Exposure Time*.
    iso
        Image *ISO*.
    black_level
        Image *Black Level*.
    white_level
        Image *White Level*.
    white_balance_multipliers
        Image white balance multipliers, usually the *As Shot Neutral*  matrix.
    """

    f_number: Optional[NDArray] = field(default_factory=lambda: None)
    exposure_time: Optional[NDArray] = field(default_factory=lambda: None)
    iso: Optional[NDArray] = field(default_factory=lambda: None)
    black_level: Optional[NDArray] = field(default_factory=lambda: None)
    white_level: Optional[NDArray] = field(default_factory=lambda: None)
    white_balance_multipliers: Optional[NDArray] = field(
        default_factory=lambda: None
    )


class Image:
    """
    Define the base object for storing an image along its path, pixel data and
    metadata needed for HDRIs generation.

    Parameters
    ----------
    path
        Image path.
    data
        Image pixel data array.
    metadata
        Image exif metadata.

    Attributes
    ----------
    -   :attr:`colour_hdri.Image.path`
    -   :attr:`colour_hdri.Image.data`
    -   :attr:`colour_hdri.Image.metadata`

    Methods
    -------
    -   :meth:`colour_hdri.Image.__init__`
    -   :meth:`colour_hdri.Image.read_data`
    -   :meth:`colour_hdri.Image.read_metadata`
    """

    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[ArrayLike] = None,
        metadata: Optional[Metadata] = None,
    ) -> None:
        self._path: Optional[str] = None
        self.path = path
        # TODO: Remove pragma when https://github.com/python/mypy/issues/3004
        # is resolved.
        self._data: Optional[NDArray] = None
        self.data = data  # type: ignore[assignment]
        self._metadata: Optional[Metadata] = None
        self.metadata = metadata

    @property
    def path(self) -> Optional[str]:
        """
        Getter and setter property for the image path.

        Parameters
        ----------
        value
            Value to set the image path with.

        Returns
        -------
        :py:data:`None` or :class:`str`
            Image path.
        """

        return self._path

    @path.setter
    def path(self, value: Optional[str]):
        """Setter for the **self._path** property."""

        if value is not None:
            assert is_string(
                value
            ), f'"path" property: "{value}" type is not "str"!'

        self._path = value

    @property
    def data(self) -> Optional[NDArray]:
        """
        Getter and setter property for the image data.

        Parameters
        ----------
        value
            Value to set the image data with.

        Returns
        -------
        :py:data:`None` or :class:`numpy.ndarray`
            Image data.
        """

        return self._data

    @data.setter
    def data(self, value: Optional[ArrayLike]):
        """Setter for the **self._data** property."""

        if value is not None:
            assert isinstance(value, (tuple, list, np.ndarray, np.matrix)), (
                f'"data" property: "{value!r}" is not a "tuple", "list", "ndarray" '
                'or "matrix" instance!'
            )

            value = as_float_array(value)

        self._data = value

    @property
    def metadata(self) -> Optional[Metadata]:
        """
        Getter and setter property for the image metadata.

        Parameters
        ----------
        value
            Value to set the image metadata with.

        Returns
        -------
        :py:data:`None` or :class:`colour_hdri.Metadata`
            Image metadata.
        """

        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Setter for the **self._metadata** property."""

        if value is not None:
            assert isinstance(
                value, Metadata
            ), f'"metadata" property: "{value}" is not a "Metadata" instance!'

        self._metadata = value

    def read_data(self, cctf_decoding: Optional[Callable] = None) -> NDArray:
        """
        Read image pixel data at :attr:`Image.path` attribute.

        Parameters
        ----------
        cctf_decoding
            Decoding colour component transfer function (Decoding CCTF) or
            electro-optical transfer function (EOTF / EOCF).

        Returns
        -------
        :class:`numpy.ndarray`
            Image pixel data.

        Raises
        ------
        ValueError
            If the image path is undefined.
        """

        if self._path is not None:
            logging.info(f'Reading "{self._path}" image.')

            data = read_image(str(self._path))
            if cctf_decoding is not None:
                data = cctf_decoding(data)

            self.data = data

            return data
        else:
            raise ValueError('The image "path" is undefined!')

    def read_metadata(self) -> Metadata:
        """
        Read image relevant exif metadata at :attr:`Image.path` attribute.

        Returns
        -------
        :class:`colour_hdri.Metadata`
            Image relevant exif metadata.

        Raises
        ------
        ValueError
            If the image path is undefined.
        """

        if self._path is not None:
            logging.info(f'Reading "{self._path}" image metadata.')

            exif_data = read_exif_tags(self._path)

            if not exif_data.get("EXIF"):
                warning(
                    f'"{self._path}" file has no "Exif" data, metadata will '
                    f"be undefined!"
                )
                self.metadata = Metadata(*[None] * 6)
                return self.metadata

            f_number = exif_data["EXIF"].get("F Number")
            if f_number is not None:
                f_number = parse_exif_number(f_number[0])

            exposure_time = exif_data["EXIF"].get("Exposure Time")
            if exposure_time is not None:
                exposure_time = parse_exif_fraction(exposure_time[0])

            iso = exif_data["EXIF"].get("ISO")
            if iso is not None:
                iso = parse_exif_number(iso[0])

            black_level = exif_data["EXIF"].get("Black Level")
            if black_level is not None:
                black_level = parse_exif_array(black_level[0])
                black_level = as_float_array(black_level) / 65535

            white_level = exif_data["EXIF"].get("White Level")
            if white_level is not None:
                white_level = parse_exif_array(white_level[0])
                white_level = as_float_array(white_level) / 65535

            white_balance_multipliers = exif_data["EXIF"].get(
                "As Shot Neutral"
            )
            if white_balance_multipliers is not None:
                white_balance_multipliers = parse_exif_array(
                    white_balance_multipliers[0]
                )
                white_balance_multipliers = (
                    as_float_array(white_balance_multipliers)
                    / white_balance_multipliers[1]
                )

            metadata = Metadata(
                f_number,
                exposure_time,
                iso,
                black_level,
                white_level,
                white_balance_multipliers,
            )

            self._metadata = metadata

            return metadata
        else:
            raise ValueError('The image "path" is undefined!')


class ImageStack(MutableSequence):
    """
    Define a convenient stack storing a sequence of images for HDRI / radiance
    images generation.

    Methods
    -------
    -   :meth:`colour_hdri.ImageStack.__init__`
    -   :meth:`colour_hdri.ImageStack.__getitem__`
    -   :meth:`colour_hdri.ImageStack.__setitem__`
    -   :meth:`colour_hdri.ImageStack.__delitem__`
    -   :meth:`colour_hdri.ImageStack.__len__`
    -   :meth:`colour_hdri.ImageStack.__getattr__`
    -   :meth:`colour_hdri.ImageStack.__setattr__`
    -   :meth:`colour_hdri.ImageStack.sort`
    -   :meth:`colour_hdri.ImageStack.insert`
    -   :meth:`colour_hdri.ImageStack.from_files`
    """

    def __init__(self) -> None:
        self._data: List = []

    def __getitem__(
        self, index: Union[Integer, slice]
    ) -> Union[Any, MutableSequence[Any]]:
        """
        Return the :class:`colour_hdri.Image` class instance at given index.

        Parameters
        ----------
        index
            :class:`colour_hdri.Image` class instance.

        Returns
        -------
        :class:`colour_hdri.Image`
            :class:`colour_hdri.Image` class instance at given index.
        """

        return self._data[index]

    def __setitem__(self, index: Union[Integer, slice], value: Any):
        """
        Set given :class:`colour_hdri.Image` class instance at given index.

        Parameters
        ----------
        index
            :class:`colour_hdri.Image` class instance index.
        value
            :class:`colour_hdri.Image` class instance to set.
        """

        self._data[index] = value

    def __delitem__(self, index: Union[Integer, slice]):
        """
        Delete the :class:`colour_hdri.Image` class instance at given index.

        Parameters
        ----------
        index
            :class:`colour_hdri.Image` class instance index.
        """

        del self._data[index]

    def __len__(self) -> Integer:
        """
        Return the :class:`colour_hdri.Image` class instances count.

        Returns
        -------
        :class:`numpy.integer`
            :class:`colour_hdri.Image` class instances count.
        """

        return len(self._data)

    def __getattr__(self, attribute: str) -> Any:
        """
        Return the value from the attribute with given name.

        Parameters
        ----------
        name
            Name of the attribute to get the value from.

        Returns
        -------
        :class:`object`
        """

        try:
            return self.__dict__[attribute]
        except KeyError:
            if hasattr(Image, attribute):
                value = [getattr(image, attribute) for image in self]
                if attribute == "data":
                    return tstack(value)
                else:
                    return tuple(value)
            # TODO: Revise then "MixinDataclassArray" is improved.
            elif attribute in [field.name for field in fields(Metadata)]:
                value = [getattr(image.metadata, attribute) for image in self]
                return as_float_array(value)
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute "
                    f"'{attribute}'"
                )

    def __setattr__(self, attribute: str, value: Any):
        """
        Set given value to the attribute with given name.

        Parameters
        ----------
        attribute
            Attribute to set the value of.
        value
            Value to set the attribute with.
        """

        if hasattr(Image, attribute):
            if attribute == "data":
                data = tsplit(value)
                for i, image in enumerate(self):
                    image.data = data[i]
            else:
                for i, image in enumerate(self):
                    setattr(image, attribute, value[i])
        elif attribute in [field.name for field in fields(Metadata)]:
            for i, image in enumerate(self):
                setattr(image.metadata, attribute, value[i])
        else:
            super().__setattr__(attribute, value)

    def insert(self, index: Integer, value: Any):
        """
        Insert given :class:`colour_hdri.Image` class instance at given index.

        Parameters
        ----------
        index
            :class:`colour_hdri.Image` class instance index.
        value
            :class:`colour_hdri.Image` class instance to set.
        """

        self._data.insert(index, value)

    def sort(self, key: Optional[Callable] = None):
        """
        Sort the underlying data structure.

        Parameters
        ----------
        key
            Function of one argument that is used to extract a comparison key
            from each data structure.
        """

        self._data = sorted(self._data, key=key)

    @staticmethod
    def from_files(
        image_files: Sequence[str], cctf_decoding: Optional[Callable] = None
    ) -> ImageStack:
        """
        Return a :class:`colour_hdri.ImageStack` instance from given image
        files.

        Parameters
        ----------
        image_files
            Image files.
        cctf_decoding
            Decoding colour component transfer function (Decoding CCTF) or
            electro-optical transfer function (EOTF / EOCF).

        Returns
        -------
        :class:`colour_hdri.ImageStack`
        """

        image_stack = ImageStack()
        for image_file in image_files:
            image = Image(image_file)
            image.read_data(cctf_decoding)
            image.read_metadata()
            image_stack.append(image)

        def luminance_average_key(image: Image) -> Optional[FloatingOrNDArray]:
            """Comparison key function."""

            metadata = cast(Metadata, image.metadata)

            f_number = metadata.f_number
            exposure_time = metadata.exposure_time
            iso = metadata.iso

            if f_number is None or exposure_time is None or iso is None:
                warning(
                    f'"{image.path}" exposure data is missing, average '
                    f"luminance sorting is inapplicable!"
                )
                return None
            else:
                return 1 / average_luminance(f_number, exposure_time, iso)

        image_stack.sort(luminance_average_key)

        return image_stack
