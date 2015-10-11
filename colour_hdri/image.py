#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np
from collections import MutableSequence
from recordclass import recordclass

from colour import tsplit, tstack


class Metadata(
    recordclass('Metadata',
                ('f_number',
                 'exposure_time',
                 'iso',
                 'black_level',
                 'white_level'))):
    def __new__(cls,
                f_number,
                exposure_time,
                iso,
                black_level=None,
                white_level=None):
        return super(Metadata, cls).__new__(
            cls, f_number, exposure_time, iso, black_level, white_level)


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
                ('"{0}" attribute: "{1}" type is not "Metadata"!').format(
                    'metadata', value))

        self.__metadata = value


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
