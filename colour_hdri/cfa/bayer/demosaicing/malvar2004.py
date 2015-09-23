#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np
from scipy.ndimage.filters import convolve
from colour import tstack

from colour_hdri.cfa.bayer.masks import masks_CFA_Bayer


def demosaicing_CFA_Bayer_Malvar2004(CFA, pattern='RGGB'):
    # http://research.microsoft.com/apps/pubs/default.aspx?id=102068

    CFA = np.asarray(CFA)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    # @formatter:off
    GR_GB = np.asarray(
        [[   0,   0,   -1,   0,   0],
         [   0,   0,    2,   0,   0],
         [  -1,   2,    4,   2,  -1],
         [   0,   0,    2,   0,   0],
         [   0,   0,   -1,   0,   0]]) / 8

    Rg_RB_Bg_BR = np.asarray(
        [[   0,   0,  0.5,   0,   0],
         [   0,  -1,    0,  -1,   0],
         [  -1,   4,    5,   4,  -1],
         [   0,  -1,    0,  -1,   0],
         [   0,   0,  0.5,   0,   0]]) / 8

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = np.asarray(
        [[   0,   0,-1.5,   0,   0],
         [   0,   2,   0,   2,   0],
         [-1.5,   0,   6,   0,-1.5],
         [   0,   2,   0,   2,   0],
         [   0,   0,-1.5,   0,   0]]) / 8
    # @formatter:on

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G = np.where(np.logical_or(R_m == 1, B_m == 1), convolve(CFA, GR_GB), G)

    RBg_RBBR = convolve(CFA, Rg_RB_Bg_BR)
    RBg_BRRB = convolve(CFA, Rg_BR_Bg_RB)
    RBgr_BBRR = convolve(CFA, Rb_BB_Br_RR)

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Red columns.
    R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    # Blue columns
    B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    return tstack((R, G, B))
