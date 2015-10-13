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

__all__ = ['normal_distribution_function',
           'hat_function',
           'weighting_function_Debevec1997']


def normal_distribution_function(a, mu=0.5, sigma=0.15):
    a = np.asarray(a)

    return np.exp(-np.power(a - mu, 2) / (2 * np.power(sigma, 2.)))


def hat_function(a):
    a = np.asarray(a)

    return 1 - (2 * a - 1) ** 12


def weighting_function_Debevec1997(a, domain_l=0.01, domain_h=0.99):
    a = np.asarray(a)

    w = np.zeros(a.shape)

    mask = np.where(a <= (domain_l + domain_h) / 2, True, False)
    w[mask] = a[mask] - domain_l
    w[~mask] = domain_h - a[~mask]
    w /= np.max(w)

    w[w < 0] = 0

    return w
