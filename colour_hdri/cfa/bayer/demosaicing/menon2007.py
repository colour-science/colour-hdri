#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy.ndimage.filters import convolve, convolve1d

from colour import tsplit, tstack

from colour_hdri.cfa.bayer.masks import masks_CFA_Bayer

_cnv_h = lambda x, y: convolve1d(x, y, mode='mirror')
_cnv_v = lambda x, y: convolve1d(x, y, mode='mirror', axis=0)


def demosaicing_CFA_Bayer_Menon2007(CFA, pattern='RGGB', refine=True):
    # http://www.danielemenon.netsons.org/pub/dfapd/dfapd.php
    CFA = np.asarray(CFA)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    h_0 = np.array([0, 0.5, 0, 0.5, 0])
    h_1 = np.array([-0.25, 0, 0.5, 0, -0.25])

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G_H = np.where(G == 0, _cnv_h(CFA, h_0) + _cnv_h(CFA, h_1), G)
    G_V = np.where(G == 0, _cnv_v(CFA, h_0) + _cnv_v(CFA, h_1), G)

    C_H = np.where(R_m == 1, R - G_H, 0)
    C_H = np.where(B_m == 1, B - G_H, C_H)

    C_V = np.where(R_m == 1, R - G_V, 0)
    C_V = np.where(B_m == 1, B - G_V, C_V)

    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode='reflect')[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode='reflect')[2:, :])

    k = np.array(
        [[0, 0, 1, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 3, 0, 3],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 1]])

    d_H = convolve(D_H, k, mode='constant')
    d_V = convolve(D_V, np.transpose(k), mode='constant')

    mask = d_V >= d_H
    G = np.where(mask, G_H, G_V)
    M = np.where(mask, 1, 0)

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)

    k_b = np.array([0.5, 0, 0.5])

    R = np.where(np.logical_and(G_m == 1, R_r == 1),
                 G + _cnv_h(R, k_b) - _cnv_h(G, k_b),
                 R)

    R = np.where(np.logical_and(G_m == 1, B_r == 1) == 1,
                 G + _cnv_v(R, k_b) - _cnv_v(G, k_b),
                 R)

    B = np.where(np.logical_and(G_m == 1, B_r == 1),
                 G + _cnv_h(B, k_b) - _cnv_h(G, k_b),
                 B)

    B = np.where(np.logical_and(G_m == 1, R_r == 1) == 1,
                 G + _cnv_v(B, k_b) - _cnv_v(G, k_b),
                 B)

    R = np.where(np.logical_and(B_r == 1, B_m == 1),
                 np.where(M == 1,
                          B + _cnv_h(R, k_b) - _cnv_h(B, k_b),
                          B + _cnv_v(R, k_b) - _cnv_v(B, k_b)),
                 R)

    B = np.where(np.logical_and(R_r == 1, R_m == 1),
                 np.where(M == 1,
                          R + _cnv_h(B, k_b) - _cnv_h(R, k_b),
                          R + _cnv_v(B, k_b) - _cnv_v(R, k_b)),
                 B)

    RGB = tstack((R, G, B))

    if refine:
        RGB = refine_Menon2007(RGB, tstack((R_m, G_m, B_m)), M)

    return RGB


demosaicing_CFA_Bayer_DFAPD = demosaicing_CFA_Bayer_Menon2007


def refine_Menon2007(RGB, RGB_m, M):
    R, G, B = tsplit(RGB)
    R_m, G_m, B_m = tsplit(RGB_m)
    M = np.asarray(M)

    # Updating of the green component.
    R_G = R - G
    B_G = B - G

    FIR = np.ones(3) / 3

    B_G_m = np.where(B_m == 1,
                     np.where(M == 1, _cnv_h(B_G, FIR), _cnv_v(B_G, FIR)), 0)
    R_G_m = np.where(R_m == 1,
                     np.where(M == 1, _cnv_h(R_G, FIR), _cnv_v(R_G, FIR)), 0)

    G = np.where(R_m == 1, R - R_G_m, G)
    G = np.where(B_m == 1, B - B_G_m, G)

    # Updating of the red and blue components in the green locations.
    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[np.newaxis]) * np.ones(R.shape)
    # Red columns.
    R_c = np.any(R_m == 1, axis=0)[np.newaxis] * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[np.newaxis]) * np.ones(B.shape)
    # Blue columns
    B_c = np.any(B_m == 1, axis=0)[np.newaxis] * np.ones(B.shape)

    R_G = R - G
    B_G = B - G

    k_b = np.array([0.5, 0, 0.5])

    R_G_m = np.where(np.logical_and(G_m == 1, B_r == 1),
                     _cnv_v(R_G, k_b),
                     R_G_m)
    R = np.where(np.logical_and(G_m == 1, B_r == 1), G + R_G_m, R)
    R_G_m = np.where(np.logical_and(G_m == 1, B_c == 1),
                     _cnv_h(R_G, k_b),
                     R_G_m)
    R = np.where(np.logical_and(G_m == 1, B_c == 1), G + R_G_m, R)

    B_G_m = np.where(np.logical_and(G_m == 1, R_r == 1),
                     _cnv_v(B_G, k_b),
                     B_G_m)
    B = np.where(np.logical_and(G_m == 1, R_r == 1), G + B_G_m, B)
    B_G_m = np.where(np.logical_and(G_m == 1, R_c == 1),
                     _cnv_h(B_G, k_b),
                     B_G_m)
    B = np.where(np.logical_and(G_m == 1, R_c == 1), G + B_G_m, B)

    # Updating of the red (blue) component in the blue (red) locations.
    R_B = R - B
    R_B_m = np.where(B_m == 1,
                     np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)), 0)
    R = np.where(B_m == 1, B + R_B_m, R)

    R_B_m = np.where(R_m == 1,
                     np.where(M == 1, _cnv_h(R_B, FIR), _cnv_v(R_B, FIR)), 0)
    B = np.where(R_m == 1, R - R_B_m, B)

    return tstack((R, G, B))
