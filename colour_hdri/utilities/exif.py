#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXIF
====

Exif data manipulation routines based on *exiftool*.
"""

from __future__ import division, unicode_literals

import logging
import os
import re
import subprocess

from colour_hdri.utilities import vivification

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['EXIF_EXECUTABLE',
           'parse_exif_data',
           'read_exif_data',
           'get_value',
           'set_value',
           'copy_tags',
           'delete_all_tags',
           'delete_backup_files',
           'update_exif_data']

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


def read_exif_data(image):
    """
    Returns given image exif image data.

    Parameters
    ----------
    image : unicode
        Image file.

    Returns
    -------
    defaultdict
        Exif data.
    """

    LOGGER.info("Reading '{0}' image exif data.".format(image))

    exif_data = vivification()
    lines = unicode(subprocess.check_output(
        [EXIF_EXECUTABLE, '-D', '-G', '-a', '-u', '-n', image]),
        'utf-8', 'ignore').split('\n')

    for line in lines:
        if not line.strip():
            continue

        group, id, tag, value = parse_exif_data(line)
        exif_data[group][tag] = (value, id)

    return exif_data


def get_value(image, tag):
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


def set_value(image, tag, value):
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


def copy_tags(source, target):
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
        target, source))

    subprocess.check_output(
        [EXIF_EXECUTABLE,
         '-overwrite_original',
         '-TagsFromFile',
         '{0}'.format(source),
         '{0}'.format(target)])

    return True


def delete_all_tags(image):
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


def delete_backup_files(directory):
    """
    Deletes *exiftool* backup image files in given directory.

    Parameters
    ----------
    directory : unicode
        Directory.

    Returns
    -------
    bool
        Definition success.
    """

    backup_image_files = map(lambda x: os.path.join(directory, x),
                             filter(lambda x: re.search('_original', x),
                                    sorted(os.listdir(directory))))

    for backup_image_file in backup_image_files:
        LOGGER.info(
            "Deleting '{0}' backup image file.".format(backup_image_file))

        os.remove(backup_image_file)

    return True


def update_exif_data(images):
    """
    Updates given images siblings exif data.

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
        success *= copy_tags(source, target)

    return success
