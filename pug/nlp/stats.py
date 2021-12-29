#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stats functions and classes useful in DOX, like `Confusion` and `cosine_distance`"""
from __future__ import print_function, division
# from past.builtins import basestring

import json
import logging
from copy import deepcopy
from itertools import product
from collections import Mapping

from scipy.optimize import minimize
import pandas as pd
from matplotlib import pyplot as plt

from pug.nlp.constant import NUMERIC_TYPES
# watch out for circular import
from pug.nlp.segmentation import stringify
from pug.nlp.util import make_dataframe, listify
from pug.nlp.util import PrettyDict
from pug.nlp.constant import INF

# from scipy import stats as scipy_stats
# import pymc

np = pd.np
logger = logging.getLogger(__name__)


def mcc_chi(mcc, num_samples):
    return np.sqrt(mcc ** 2 * num_samples) if mcc is not None and num_samples else 0
phi2chi = mcc_chi


def random_split(df, first_portion=.1):
    """Randomly sample Dataframe rows. Typically for testset/training set cross-valication

    random_split(pd.DataFrame())
    """
    mask = np.random.binomial(1, first_portion, size=len(df)).astype(bool)
    return df[mask].copy(), df[~mask].copy()


def safe_div(a, b, inf=INF):
    """Safely divide by zero and inf and nan and produce floating point result
    Numerical results are equivalent to `from __future__ import division`
    Args:
      a (float or int): numerator
      b (float or int): denominator
      inf (float or object): value to return in place of division by zero errors
      none (float or object): value to return in place of TypeErrors (division by None)
    Returns:
      dividend:
        b if b is None or NaN
        a if a is None or NaN
        inf (specified by caller, default is np.inf) if b is 0
        pd.NaT if b or a is pd.NaT
    >>> safe_div(84, 2)
    42.0
    >>> safe_div(-85, 2)
    -42.5
    >>> safe_div(42, 0)
    inf
    >>> safe_div(-42, 0)
    inf
    >>> safe_div(1, 0.)
    inf
    >>> safe_div(1, 0., inf="custom OutOfBounds str")
    'custom OutOfBounds str'
    >>> safe_div(np.inf, np.inf)
    nan
    >>> safe_div(1e200, np.inf)
    0.0
    >>> safe_div(np.inf, np.nan)
    nan
    >>> safe_div(np.nan, 0)
    inf
    >>> repr(safe_div(99.99, None))
    'None'
    >>> safe_div(42, object())
    <object object at 0x...>
    """
    try:
        return 1. * a / b
    except ZeroDivisionError:
        return inf
    except TypeError:
        try:
            1. / b
        except TypeError:
            return b
        return a
    return 1. * a / b


def mcc_chi(mcc, num_samples):
    """Return the equivalent Chi value for a given MCC (correlation) value
    >>> round(mcc_chi(0.5, 100), 3)
    5.0
    """
    return np.sqrt(mcc ** 2. * num_samples) if mcc is not None and num_samples else 0.
phi2chi = mcc_chi


def mcc_chi_squared(mcc, num_samples):
    """Return the equivalent Chi-Squared value for a given MCC (correlation) value
    Chi is a cumulative value on the horizontal axes (z-score) and represents confidence in c