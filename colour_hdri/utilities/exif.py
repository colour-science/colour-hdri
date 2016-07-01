#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXIF Manipulation
=================

Exif data manipulation routines based on *exiftool*:

-   :func:`parse_exif_data`
-   :func:`read_exif_tags`
-   :func:`copy_exif_tags`
-   :func:`update_exif_tags`
-   :func:`delete_exif_tags`
-   :func:`read_exif_tag`
-   :func:`write_exif_tag`
"""

from __future__ import division, unicode_literals

import logging
import re
import subprocess

from colour_hdri.utilities import vivification

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['EXIF_EXECUTABLE',
           'parse_exif_data',
           'read_exif_tags',
           'copy_exif_tags',
           'update_exif_tags',
           'delete_exif_tags',
           'read_exif_tag',
           'write_exif_tag']

LOGGER = logging.getLogger(__name__)

EXIF_EXECUTABLE = 'exiftool'


def parse_exif_data(data):
    """
    Parses given exif data output from *exiftool*.

    Parameters
    ----------
    data : unicode
        Exif data.

    Returns
    -------
    list
        Parsed exif data.
    """

    search = re.search(
        r'\[(?P<group>\w+)\]\s*(?P<id>(\d+|-))?(?P<tag>.*?):(?P<value>.*$)',
        data)

    return map(lambda x: x.strip() if x is not None else x,
               (search.group('group'),
                search.group('id'),
                search.group('tag'),
                search.group('value')))


def read_exif_tags(image):
    """
    Returns given image exif image tags.

    Parameters
    ----------
    image : unicode
        Image file.

    Returns
    -------
    defaultdict
        Exif tags.
    """

    LOGGER.info("Reading '{0}' image exif data.".format(image))

    exif_tags = vivification()
    lines = unicode(subprocess.check_output(
        [EXIF_EXECUTABLE, '-D', '-G', '-a', '-u', '-n', image]),
        'utf-8', 'ignore').split('\n')

    for line in lines:
        if not line.strip():
            continue

        group, identifier, tag, value = parse_exif_data(line)
        exif_tags[group][tag] = (value, identifier)

    return exif_tags


def copy_exif_tags(source, target):
    """
    Copies given source image file exif tag to given image target.

    Parameters
    ----------
    source : unicode
        Source image file.
    target : unicode
        Target image file.

    Returns
    -------
    bool
        Definition success.
    """

    LOGGER.info("Copying '{0}' file exif data to '{1}' file.".format(
        source, target))

    subprocess.check_output(
        [EXIF_EXECUTABLE,
         '-overwrite_original',
         '-TagsFromFile',
         '{0}'.format(source),
         '{0}'.format(target)])

    return True


def update_exif_tags(images):
    """
    Updates given images siblings images pairs exif tags.

    Parameters
    ----------
    images : list
        Image files to update.

    Returns
    -------
    bool
        Definition success.
    """

    success = True
    for (source, target) in images:
        success *= copy_exif_tags(source, target)

    return success


def delete_exif_tags(image):
    """
    Deletes all given image exif tags.

    Parameters
    ----------
    image : unicode
        Image file.

    Returns
    -------
    bool
        Definition success.
    """

    LOGGER.info("Deleting '{0}' image exif tags.".format(image))

    subprocess.check_output(
        [EXIF_EXECUTABLE, '-overwrite_original', '-all=', image])

    return True


def read_exif_tag(image, tag):
    """
    Returns given image exif tag value.

    Parameters
    ----------
    image : unicode
        Image file.
    tag : unicode
        Tag.

    Returns
    -------
    unicode
        Tag value.
    """

    value = unicode(subprocess.check_output(
        [EXIF_EXECUTABLE, '-{0}'.format(tag), image]),
        'utf-8', 'ignore').split(':').pop().strip()

    LOGGER.info("Reading '{0}' image '{1}' exif tag value: '{2}'".format(
        image, tag, value))

    return value


def write_exif_tag(image, tag, value):
    """
    Sets given image exif tag value.

    Parameters
    ----------
    image : unicode
        Image file.
    tag : unicode
        Tag.
    value : unicode
        Value.

    Returns
    -------
    bool
        Definition success.
    """

    LOGGER.info("Writing '{0}' image '{1}' exif tag with '{2}' value.".format(
        image, tag, value))

    subprocess.check_output(
        [EXIF_EXECUTABLE, '-overwrite_original', '-{0}={1}'.format(tag, value),
         image])

    return True
