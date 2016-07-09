#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Utilities
================

Defines common utilities objects that don't fall in any specific category.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import re

from collections import defaultdict

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['linear_conversion',
           'vivification',
           'vivified_to_dict',
           'path_exists',
           'filter_files']


def linear_conversion(a, old_range, new_range):
    """
    Performs a simple linear conversion of given array between the old and new
    ranges.

    Parameters
    ----------
    a : array_like
        Array to perform the linear conversion onto.
    old_range : array_like
        Old range.
    new_range : array_like
        New range.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> linear_conversion(a, np.array([0, 1]), np.array([1, 10]))
    array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])
    """

    a = np.asarray(a)

    in_min, in_max = old_range
    out_min, out_max = new_range

    return (((a - in_min) / (in_max - in_min)) *
            (out_max - out_min) + out_min)


def vivification():
    """
    Implements supports for vivification of the underlying dict like
    data-structure, magical!

    Returns
    -------
    defaultdict

    Examples
    --------
    >>> vivified = vivification()
    >>> vivified['my']['attribute'] = 1
    >>> vivified['my']  # doctest: +ELLIPSIS
    defaultdict(<function vivification at 0x...>, {u'attribute': 1})
    >>> vivified['my']['attribute']
    1
    """

    return defaultdict(vivification)


def vivified_to_dict(vivified):
    """
    Converts given vivified data-structure to dictionary.

    Parameters
    ----------
    vivified : defaultdict
        Vivified data-structure.

    Returns
    -------
    dict

    Examples
    --------
    >>> vivified = vivification()
    >>> vivified['my']['attribute'] = 1
    >>> vivified_to_dict(vivified)
    {u'my': {u'attribute': 1}}
    """

    if isinstance(vivified, defaultdict):
        vivified = {key: vivified_to_dict(value)
                    for key, value in vivified.iteritems()}
    return vivified


def path_exists(path):
    """
    Returns if given path exists.

    Parameters
    ----------
    path : unicode
        Path to check the existence.

    Returns
    -------
    bool

    Examples
    --------
    >>> path_exists(__file__)
    True
    >>> path_exists('')
    False
    """

    if not path:
        return False
    else:
        return os.path.exists(path)


def filter_files(directory, extensions):
    """
    Filters given directory for files matching given extensions.

    Parameters
    ----------
    directory : unicode
        Directory to filter.
    extensions : tuple or list
        Extensions to filter on.

    Returns
    -------
    list
        Filtered files.
    """

    return map(lambda x: os.path.join(directory, x),
               filter(lambda x: re.search('\.({0})$'.format(
                   '|'.join(extensions)), x), sorted(os.listdir(directory))))
