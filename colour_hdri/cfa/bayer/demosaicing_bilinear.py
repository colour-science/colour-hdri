#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np
from scipy.ndimage.filters import convolve
from colour import tstack

from colour_hdri.cfa.bayer.masks import masks_CFA_Bayer


def demosaicing_bilinear_CFA_Bayer(CFA, pattern='RGGB'):
    # https://hal.inria.fr/hal-00683233/PDF/AEIP_SOUMIS.pdf
    CFA = np.asarray(CFA)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    H_G = np.asarray([[0, 1, 0],
                      [1, 4, 1],
                      [0, 1, 0]]) / 4

    H_RB = np.asarray([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 4

    R = convolve(CFA * R_m, H_RB)
    G = convolve(CFA * G_m, H_G)
    B = convolve(CFA * B_m, H_RB)

    return tstack((R, G, B))
