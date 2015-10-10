#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

from collections import namedtuple

Image = namedtuple('Image', ('path', 'data', 'exif_data'))
