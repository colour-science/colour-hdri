#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXIF
====

Exif data manipulation routines based on **exiftool**.
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

__all__ = ['LOGGER'
           'EXIF_EXECUTABLE',
           'parse_exif_data',
           'get_exif_data',
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
    Parses given exif data.

    :param data: Exif data.
    :type data: unicode
    :return: Parsed exif data.
    :rtype: tuple
    """

    search = re.search(
        r'\[(?P<group>\w+)\]\s*(?P<id>(\d+|-))?(?P<tag>.*?):(?P<value>.*$)',
        data)

    return map(lambda x: x.strip() if x is not None else x,
               (search.group('group'),
                search.group('id'),
                search.group('tag'),
                search.group('value')))


def get_exif_data(file):
    """
    Returns given file exif file data.

    :param file: File.
    :type file: unicode
    :return: Exif data.
    :rtype: dict
    """

    LOGGER.info("Reading '{0}' file exif data.".format(file))

    exif_data = vivification()
    lines = unicode(subprocess.check_output(
        [EXIF_EXECUTABLE, '-D', '-G', '-a', '-u', '-n', file]),
        'utf-8', 'ignore').split('\n')

    for line in lines:
        if not line.strip():
            continue

        group, id, tag, value = parse_exif_data(line)
        exif_data[group][tag] = (value, id)

    return exif_data


def get_value(file, tag):
    """
    Returns given file exif tag value.

    :param file: File.
    :type file: unicode
    :param tag: Tag.
    :type tag: unicode
    :return: Tag value.
    :rtype: unicode
    """

    value = unicode(subprocess.check_output(
        [EXIF_EXECUTABLE, '-{0}'.format(tag), file]),
        'utf-8', 'ignore').split(':').pop().strip()

    LOGGER.info("Reading '{0}' file '{1}' exif tag value: '{2}'".format(
        file, tag, value))

    return value


def set_value(file, tag, value):
    """
    Sets given file exif tag value.

    :param file: File.
    :type file: unicode
    :param tag: Tag.
    :type tag: unicode
    :param value: Value.
    :type value: unicode
    :return: Definition success.
    :rtype: bool
    """

    LOGGER.info("Writing '{0}' file '{1}' exif tag with '{2}' value.".format(
        file, tag, value))

    subprocess.check_output(
        [EXIF_EXECUTABLE, '-overwrite_original', '-{0}={1}'.format(tag, value),
         file])

    return True


def copy_tags(source, target):
    """
    Copies given source file exif tag to given target.

    :param source: Source file.
    :type source: unicode
    :param target: Target file.
    :type target: unicode
    :return: Definition success.
    :rtype: bool
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


def delete_all_tags(file):
    """
    Deletes all given file exif tags.

    :param file: File.
    :type file: unicode
    :return: Definition success.
    :rtype: bool
    """

    LOGGER.info("Deleting '{0}' file exif tags.".format(file))

    subprocess.check_output([EXIF_EXECUTABLE, '-overwrite_original', '-all=', file])

    return True


def delete_backup_files(directory):
    """
    Deletes 'exiftool' backup files in given directory.

    :param directory: Directory.
    :type directory: unicode
    :return: Definition success.
    :rtype: bool
    """

    backup_files = map(lambda x: os.path.join(directory, x),
                       filter(lambda x: re.search('_original', x),
                              sorted(os.listdir(directory))))

    for backup_file in backup_files:
        LOGGER.info("Deleting '{0}' backup file.".format(backup_file))

        os.remove(backup_file)

    return True


def update_exif_data(files):
    """
    Updates given files siblings exif data.
    :param files: Files to update.
    :type files: list
    :return: Definition success.
    :rtype: bool
    """

    success = True
    for (source, target) in files:
        success *= copy_tags(source, target)

    return success
