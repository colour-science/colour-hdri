#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np


def exposure_value(aperture, shutter_speed, iso):
    N = np.asarray(aperture)
    t = np.asarray(shutter_speed)
    S = np.asarray(iso)

    EV = np.log2(N ** 2) + np.log2(1 / t) - np.log2(100 / S)

    return EV


def average_luminance(aperture, shutter_speed, iso, k=12.5):
    N = np.asarray(aperture)
    t = np.asarray(shutter_speed)
    S = np.asarray(iso)

    L = (S * t) / (k * N ** 2)

    return L
