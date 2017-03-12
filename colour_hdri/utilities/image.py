#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Data & Metadata Utilities
===============================

Defines various image data and metadata utilities classes:

-   :class:`Metadata`
-   :class:`Image`
-   :class:`ImageStack`
"""

from __future__ import division, unicode_literals

import logging
import numpy as np
from collections import MutableSequence
from recordclass import recordclass

from colour import is_string, read_image, tsplit, tstack

from colour_hdri.utilities.exif import (
    parse_exif_array,
    parse_exif_fraction,
    parse_exif_numeric,
    read_exif_tags)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Metadata',
           'Image',
           'ImageStack']

LOGGER = logging.getLogger(__name__)


class Metadata(
    recordclass('Metadata',
                ('f_number',
                 'exposure_time',
                 'iso',
                 'black_level',
                 'white_level',
                 'white_balance_multipliers'))):
    """
    Defines the base object for storing exif metadata relevant to
    HDRI / radiance image generation.

    Parameters
    ----------
    f_number : array_like
        Image *FNumber*.
    exposure_time : array_like
        Image *Exposure Time*.
    iso : array_like
        Image *ISO*.
    black_level : array_like
        Image *Black Level*.
    white_level : array_like
        Image *White Level*.
    white_balance_multipliers : array_like
        Image white balance multipliers, usually the *As Shot Neutral*  matrix.
    """

    def __new__(cls,
                f_number,
                exposure_time,
                iso,
                black_level=None,
                white_level=None,
                white_balance_multipliers=None):
        return super(Metadata, cls).__new__(
            cls,
            f_number,
            exposure_time,
            iso,
            black_level,
            white_level,
            white_balance_multipliers)


class Image(object):
    """
    Defines the base object for storing an image along its path, pixel data and
    metadata needed for HDRI / radiance images generation.

    Parameters
    ----------
    path : unicode, optional
        Image path.
    data : array_like, optional
        Image pixel data array.
    metadata : Metadata, optional
        Image exif metadata.

    Attributes
    ----------
    path
    data
    metadata

    Methods
    -------
    read_data
    read_metadata
    """

    def __init__(self, path=None, data=None, metadata=None):
        self._path = None
        self.path = path
        self._data = None
        self.data = data
        self._metadata = None
        self.metadata = metadata

    @property
    def path(self):
        """
        Property for **self._path** private attribute.

        Returns
        -------
        unicode
            self._path.
        """

        return self._path

    @path.setter
    def path(self, value):
        """
        Setter for **self._path** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a '
                 '"string" like object!').format('path', value))

        self._path = value

    @property
    def data(self):
        """
        Property for **self._data** private attribute.

        Returns
        -------
        unicode
            self._data.
        """

        return self._data

    @data.setter
    def data(self, value):
        """
        Setter for **self._data** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, (tuple, list, np.ndarray, np.matrix)), (
                ('"{0}" attribute: "{1}" is not a "tuple", "list", "ndarray" '
                 'or "matrix" instance!').format('data', value))

        self._data = np.asarray(value)

    @property
    def metadata(self):
        """
        Property for **self._metadata** private attribute.

        Returns
        -------
        unicode
            self._metadata.
        """

        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """
        Setter for **self._metadata** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, Metadata), (
                '"{0}" attribute: "{1}" is not a "Metadata" instance!'.format(
                    'metadata', value))

        self._metadata = value

    def read_data(self):
        """
        Reads image pixel data at :attr:`Image.path` attribute.

        Returns
        -------
        ndarray
            Image pixel data.
        """

        LOGGER.info('Reading "{0}" image.'.format(self._path))
        self.data = read_image(str(self._path))

        return self.data

    def read_metadata(self):
        """
        Reads image relevant exif metadata at :attr:`Image.path` attribute.

        Returns
        -------
        Metadata
            Image relevant exif metadata.
        """

        LOGGER.info('Reading "{0}" image metadata.'.format(self._path))
        exif_data = read_exif_tags(self._path)
        if not exif_data.get('EXIF'):
            raise RuntimeError(
                '"{0}" file has no "Exif" data!'.format(self._path))

        f_number = exif_data['EXIF'].get('F Number')
        if f_number is not None:
            f_number = parse_exif_numeric(f_number[0])

        exposure_time = exif_data['EXIF'].get('Exposure Time')
        if exposure_time is not None:
            exposure_time = parse_exif_fraction(exposure_time[0])

        iso = exif_data['EXIF'].get('ISO')
        if iso is not None:
            iso = parse_exif_numeric(iso[0])

        black_level = exif_data['EXIF'].get('Black Level')
        if black_level is not None:
            black_level = parse_exif_array(black_level[0])
            black_level = np.asarray(black_level) / 65535

        white_level = exif_data['EXIF'].get('White Level')
        if white_level is not None:
            white_level = parse_exif_numeric(white_level[0]) / 65535

        white_balance_multipliers = exif_data['EXIF'].get('As Shot Neutral')
        if white_balance_multipliers is not None:
            white_balance_multipliers = parse_exif_array(
                white_balance_multipliers[0])
            white_balance_multipliers = np.asarray(
                white_balance_multipliers) / white_balance_multipliers[1]

        self.metadata = Metadata(
            f_number,
            exposure_time,
            iso,
            black_level,
            white_level,
            white_balance_multipliers)

        return self.metadata


class ImageStack(MutableSequence):
    """
    Defines a convenient stack storing a sequence of images for HDRI / radiance
    images generation.

    Methods
    -------
    ImageStack
    __init__
    __getitem__
    __setitem__
    __delitem__
    __len__
    __getattr__
    __setattr__
    insert
    from_files
    """

    def __init__(self):
        self._list = []

    def __getitem__(self, index):
        """
        Reimplements the :meth:`MutableSequence.__getitem__` method.

        Parameters
        ----------
        index : int
            Item index.

        Returns
        -------
        Image
            Item at given index.
        """

        return self._list[index]

    def __setitem__(self, index, value):
        """
        Reimplements the :meth:`MutableSequence.__setitem__` method.

        Parameters
        ----------
        index : int
            Item index.
        value : int
            Item value.
        """

        self._list[index] = value

    def __delitem__(self, index):
        """
        Reimplements the :meth:`MutableSequence.__delitem__` method.

        Parameters
        ----------
        index : int
            Item index.
        """

        del self._list[index]

    def __len__(self):
        """
        Reimplements the :meth:`MutableSequence.__len__` method.
        """

        return len(self._list)

    def __getattr__(self, attribute):
        """
        Reimplements the :meth:`MutableSequence.__getattr__` method.

        Parameters
        ----------
        attribute : unicode
            Attribute to retrieve the value.

        Returns
        -------
        object
            Attribute value.
        """

        try:
            return self.__dict__[attribute]
        except KeyError:
            if hasattr(Image, attribute):
                value = [getattr(image, attribute) for image in self]
                if attribute == 'data':
                    return tstack(value)
                else:
                    return tuple(value)
            elif hasattr(Metadata, attribute):
                value = [getattr(image.metadata, attribute) for image in self]
                return np.asarray(value)
            else:
                raise AttributeError(
                    "'{0}' object has no attribute '{1}'".format(
                        self.__class__.__name__, attribute))

    def __setattr__(self, attribute, value):
        """
        Reimplements the :meth:`MutableSequence.__getattr__` method.

        Parameters
        ----------
        attribute : unicode
            Attribute to set the value.
        value : object
            Value to set.
        """

        if hasattr(Image, attribute):
            if attribute == 'data':
                data = tsplit(value)
                for i, image in enumerate(self):
                    image.data = data[i]
            else:
                for i, image in enumerate(self):
                    setattr(image, attribute, value[i])
        elif hasattr(Metadata, attribute):
            for i, image in enumerate(self):
                setattr(image.metadata, attribute, value[i])
        else:
            super(ImageStack, self).__setattr__(attribute, value)

    def insert(self, index, value):
        """
        Reimplements the :meth:`MutableSequence.insert` method.

        Parameters
        ----------
        index : int
            Item index.
        value : object
            Item value.
        """

        self._list.insert(index, value)

    @staticmethod
    def from_files(image_files):
        """
        Returns a :class:`ImageStack` instance with given image files.

        Parameters
        ----------
        image_files : array_like
            Image files.

        Returns
        -------
        ImageStack
        """

        image_stack = ImageStack()
        for image_file in image_files:
            image = Image(image_file)
            image.read_data()
            image.read_metadata()
            image_stack.append(image)

        return image_stack
