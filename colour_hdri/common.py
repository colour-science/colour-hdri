#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

from collections import defaultdict


def vivication():
    return defaultdict(vivication)


class Structure(dict):
    """
    Defines an object similar to C/C++ structured type.
    Usage:
        >>> person = Structure(first_name="Doe", last_name="John", gender="male")
        >>> person.first_name
        'Doe'
        >>> person.keys()
        ['gender', 'first_name', 'last_name']
        >>> person["gender"]
        'male'
        >>> del(person["gender"])
        >>> person["gender"]
        Traceback (most recent call last):
          File "<console>", line 1, in <module>
        KeyError: 'gender'
        >>> person.gender
        Traceback (most recent call last):
          File "<console>", line 1, in <module>
        AttributeError: 'Structure' object has no attribute 'gender'
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the class.
        :param \*args: Arguments.
        :type \*args: \*
        :param \*\*kwargs: Key / Value pairs.
        :type \*\*kwargs: dict
        """

        dict.__init__(self, **kwargs)
        self.__dict__.update(**kwargs)

    def __getattr__(self, attribute):
        """
        Returns given attribute value.
        :return: Attribute value.
        :rtype: object
        """

        try:
            return dict.__getitem__(self, attribute)
        except KeyError:
            raise AttributeError("'{0}' object has no attribute '{1}'".format(
                self.__class__.__name__, attribute))

    def __setattr__(self, attribute, value):
        """
        Sets both key and sibling attribute with given value.
        :param attribute: Attribute.
        :type attribute: object
        :param value: Value.
        :type value: object
        """

        dict.__setitem__(self, attribute, value)
        object.__setattr__(self, attribute, value)

    __setitem__ = __setattr__

    def __delattr__(self, attribute):
        """
        Deletes both key and sibling attribute.
        :param attribute: Attribute.
        :type attribute: object
        """

        dict.__delitem__(self, attribute)
        object.__delattr__(self, attribute)

    __delitem__ = __delattr__

    def update(self, *args, **kwargs):
        """
        Reimplements the :meth:`Dict.update` method.
        :param \*args: Arguments.
        :type \*args: \*
        :param \*\*kwargs: Keywords arguments.
        :type \*\*kwargs: \*\*
        """

        dict.update(self, *args, **kwargs)
        self.__dict__.update(*args, **kwargs)
