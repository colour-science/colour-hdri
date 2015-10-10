#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

ExposureData = namedtuple('ExposureData',
                          ('aperture',
                           'shutter_speed',
                           'iso',
                           'black_level',
                           'white_level'))


class Image(object):
    def __init__(self, path=None, pixel_data=None, exposure_data=None):
        self.__path = None
        self.path = path
        self.__pixel_data = None
        self.pixel_data = pixel_data
        self.__exposure_data = None
        self.exposure_data = exposure_data

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
    def pixel_data(self):
        """
        Property for **self.__pixel_data** private attribute.

        Returns
        -------
        unicode
            self.__pixel_data.
        """

        return self.__pixel_data

    @pixel_data.setter
    def pixel_data(self, value):
        """
        Setter for **self.__pixel_data** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert type(value) in (tuple, list, np.ndarray, np.matrix), (
                ('"{0}" attribute: "{1}" type is not "tuple", "list", '
                 '"ndarray" or "matrix"!').format('pixel_data', value))

        self.__pixel_data = np.asarray(value)

    @property
    def exposure_data(self):
        """
        Property for **self.__exposure_data** private attribute.

        Returns
        -------
        unicode
            self.__exposure_data.
        """

        return self.__exposure_data

    @exposure_data.setter
    def exposure_data(self, value):
        """
        Setter for **self.__exposure_data** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert type(value) is ExposureData, (
                ('"{0}" attribute: "{1}" type is not "ExposureData"!').format(
                    'exposure_data', value))

        self.__exposure_data = value
