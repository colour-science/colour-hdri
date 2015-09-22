#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour import tsplit

from colour_hdri.cfa.bayer.masks import masks_CFA_Bayer


def mosaicing_CFA_Bayer(RGB, pattern='RGGB'):
    RGB = np.asarray(RGB)

    R, G, B = tsplit(RGB)
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2], pattern)

    CFA = R * R_m + G * G_m + B * B_m

    return CFA
