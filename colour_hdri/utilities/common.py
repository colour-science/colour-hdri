#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Utilities
================

Defines common utilities objects that don't fall in any specific category.
"""

from __future__ import division, unicode_literals

import os
import re

from collections import defaultdict

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['vivification',
           'vivified_to_dict',
           'path_exists',
           'filter_files']


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
               filter(lambda x: re.search('{0}$'.format(
                   '|'.join(extensions)), x), sorted(os.listdir(directory))))
