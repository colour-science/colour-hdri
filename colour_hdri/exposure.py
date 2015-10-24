#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['exposure_value',
           'adjust_exposure',
           'average_luminance']


def exposure_value(f_number, exposure_time, iso):
    N = np.asarray(f_number)
    t = np.asarray(exposure_time)
    S = np.asarray(iso)

    EV = np.log2(N ** 2) + np.log2(1 / t) - np.log2(100 / S)

    return EV


def adjust_exposure(a, ev):
    a = np.asarray(a)

    return a * pow(2, ev)


def average_luminance(f_number, exposure_time, iso, k=12.5):
    N = np.asarray(f_number)
    t = np.asarray(exposure_time)
    S = np.asarray(iso)

    L = (S * t) / (k * N ** 2)

    return L
