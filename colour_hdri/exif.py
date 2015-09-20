#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXIF
====

Exif data manipulation routines based on **exiftool**.
"""

from __future__ import division, unicode_literals

import os
import re
import subprocess
from collections import defaultdict

__author__ = 'Thomas Mansencal'
__copyright__ = 'Copyright (C) 2013 - 2014 - Thomas Mansencal'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Thomas Mansencal'
__email__ = 'thomas.mansencal@gmail.com'
__status__ = 'Production'

__all__ = ['EXIF_TOOL',
           'parse_exif_data',
           'get_exif_data',
           'get_value',
           'set_value',
           'copy_tags',
           'delete_all_tags',
           'delete_backup_files']

EXIF_TOOL = '/usr/local/bin/exiftool'


def vivication():
    return defaultdict(vivication)


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

    exif_data = vivication()
    lines = unicode(subprocess.check_output(
        [EXIF_TOOL, '-D', '-G', '-a', '-u', '-n', file]),
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
        [EXIF_TOOL, '-{0}'.format(tag), file]),
        'utf-8', 'ignore').split(':').pop().strip()
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

    subprocess.check_output(
        [EXIF_TOOL, '-overwrite_original', '-{0}={1}'.format(tag, value),
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

    subprocess.check_output(
        [EXIF_TOOL, '-overwrite_original', '-TagsFromFile', source, target])
    return True


def delete_all_tags(file):
    """
    Deletes all given file exif tags.

    :param file: File.
    :type file: unicode
    :return: Definition success.
    :rtype: bool
    """

    subprocess.check_output([EXIF_TOOL, '-overwrite_original', '-all=', file])
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
        os.remove(backup_file)
    return True
