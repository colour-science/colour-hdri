#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import logging
import numpy as np
from collections import MutableSequence
from fractions import Fraction
from recordclass import recordclass

from colour import read_image, tsplit, tstack

from colour_hdri.exif import get_exif_data

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
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
    def __init__(self, path=None, data=None, metadata=None):
        self.__path = None
        self.path = path
        self.__data = None
        self.data = data
        self.__metadata = None
        self.metadata = metadata

    @property
    def path(self):
        """
        Property for **self.__path** private attribute.

        Returns
        -------
        unicode
            self.__path.
        """

        return self.__path

    @path.setter
    def path(self, value):
        """
        Setter for **self.__path** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert type(value) in (str, unicode), (
                ('"{0}" attribute: "{1}" type is not "str" or '
                 '"unicode"!').format('path', value))

        self.__path = value

    @property
    def data(self):
        """
        Property for **self.__data** private attribute.

        Returns
        -------
        unicode
            self.__data.
        """

        return self.__data

    @data.setter
    def data(self, value):
        """
        Setter for **self.__data** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert type(value) in (tuple, list, np.ndarray, np.matrix), (
                ('"{0}" attribute: "{1}" type is not "tuple", "list", '
                 '"ndarray" or "matrix"!').format('data', value))

        self.__data = np.asarray(value)

    @property
    def metadata(self):
        """
        Property for **self.__metadata** private attribute.

        Returns
        -------
        unicode
            self.__metadata.
        """

        return self.__metadata

    @metadata.setter
    def metadata(self, value):
        """
        Setter for **self.__metadata** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert type(value) is Metadata, (
                '"{0}" attribute: "{1}" type is not "Metadata"!'.format(
                    'metadata', value))

        self.__metadata = value

    def read_data(self):
        LOGGER.info('Reading "{0}" image.'.format(self.__path))
        self.data = read_image(str(self.__path))

    def read_metadata(self):
        LOGGER.info('Reading "{0}" image metadata.'.format(self.__path))
        exif_data = get_exif_data(self.__path)
        if not exif_data.get('EXIF'):
            raise RuntimeError(
                '"{0}" file has no "Exif" data!'.format(self.__path))

        f_number = exif_data['EXIF'].get('F Number')
        if f_number is not None:
            f_number = float(f_number[0])

        exposure_time = exif_data['EXIF'].get('Exposure Time')
        if exposure_time is not None:
            exposure_time = float(Fraction(exposure_time[0]))

        iso = exif_data['EXIF'].get('ISO')
        if iso is not None:
            iso = float(iso[0])

        black_level = exif_data['EXIF'].get('Black Level')
        if black_level is not None:
            black_level = map(float, black_level[0].split())
            black_level = np.asarray(black_level) / 65535

        white_level = exif_data['EXIF'].get('White Level')
        if white_level is not None:
            white_level = float(white_level[0]) / 65535

        white_balance_multipliers = exif_data['EXIF'].get('As Shot Neutral')
        if white_balance_multipliers is not None:
            white_balance_multipliers = map(
                float, white_balance_multipliers[0].split())
            white_balance_multipliers = np.asarray(
                white_balance_multipliers) / white_balance_multipliers[1]

        self.metadata = Metadata(
            f_number,
            exposure_time,
            iso,
            black_level,
            white_level,
            white_balance_multipliers)


class ImageStack(MutableSequence):
    def __init__(self):
        self.__list = []

    def __getitem__(self, index):
        return self.__list[index]

    def __setitem__(self, index, value):
        self.__list[index] = value

    def __delitem__(self, index):
        del self.__list[index]

    def __len__(self):
        return len(self.__list)

    def insert(self, index, value):
        self.__list.insert(index, value)

    def __getattr__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            if hasattr(Image, item):
                value = [getattr(image, item) for image in self]
                if item == 'data':
                    return tstack(value)
                else:
                    return tuple(value)
            elif hasattr(Metadata, item):
                value = [getattr(image.metadata, item) for image in self]
                return np.asarray(value)
            else:
                raise AttributeError(
                    "'{0}' object has no item '{1}'".format(
                        self.__class__.__name__, item))

    def __setattr__(self, key, value):
        if hasattr(Image, key):
            if key == 'data':
                data = tsplit(value)
                for i, image in enumerate(self):
                    image.data = data[i]
            else:
                for i, image in enumerate(self):
                    setattr(image, key, value[i])
        elif hasattr(Metadata, key):
            for i, image in enumerate(self):
                setattr(image.metadata, key, value[i])
        else:
            super(ImageStack, self).__setattr__(key, value)
