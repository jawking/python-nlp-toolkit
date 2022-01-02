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
    Chi is a cumulative value on the horizontal axes (z-score) and represents confidence in correlation
    (like the inverse of P-value) or statistical significance.
    Assumes that the random variable being measured is continuous rather than discrete.
    Otherwise num_samples should be the number of degrees of freedom: (num_possible_states - 1) ** 2
    >>> round(mcc_chi_squared(0.5, 100), 3)
    25.0
    """
    return mcc_chi(mcc, num_samples) ** 2.


def tptnfpfn_chi(*args, **kwargs):
    """Calculate Chi from True Positive (tp), True Negative (tn), False Positive/Negative counts.
    Assumes that the random variable being measured is continuous rather than discrete.
    Reference:
      https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    >>> round(tptnfpfn_chi(1000, 2000, 30, 40))
    2765.0
    """
    tp, tn, fp, fn = args_tptnfpfn(*args, **kwargs)
    return tptnfpfn_mcc(tp=tp, tn=tn, fp=fp, fn=fn) ** 2. * (tp + tn + fp + fn)


def args_tptnfpfn(*args, **kwargs):
    """Convert kwargs for tp, tn, fp, fn to ordered tuple of args
    If a single tuple/list is passed as the first arg, it is assumed to be the desired tuple of args
    >>> args_tptnfpfn(1, 2, 3, 4)
    (1, 2, 3, 4)
    >>> args_tptnfpfn((1, 2, 3, 4))
    (1, 2, 3, 4)
    >>> args_tptnfpfn([1, 2, 3, 4])
    (1, 2, 3, 4)
    >>> args_tptnfpfn(3, 4, tp=1, tn=2)
    (1, 2, 3, 4)
    >>> args_tptnfpfn(tp=1, tn=2)
    (1, 2, 0, 0)
    >>> args_tptnfpfn(tp=1, tn=2, fp=3, fn=4)
    (1, 2, 3, 4)
    >>> args_tptnfpfn(1)
    (1, 0, 0, 0)
    """
    if len(args) == 4:
        tp, tn, fp, fn = args
    elif len(kwargs) == 0:
        if len(args) == 1:
            args = listify(args[0])
        tp, tn, fp, fn = list(list(args) + [0] * (4 - len(args)))
    else:
        args = list(args)
        tp = kwargs['tp'] if 'tp' in kwargs else args.pop(0) if len(args) else 0
        tn = kwargs['tn'] if 'tn' in kwargs else args.pop(0) if len(args) else 0
        fp = kwargs['fp'] if 'fp' in kwargs else args.pop(0) if len(args) else 0
        fn = kwargs['fn'] if 'fn' in kwargs else args.pop(0) if len(args) else 0
    return tp, tn, fp, fn


def tptnfpfn_mcc(*args, **kwargs):
    tp, tn, fp, fn = args_tptnfpfn(*args, **kwargs)
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def dataframe_tptnfpfn(df, pos_label=True, labels=None):
    """Count the True Pos, True Neg, False Pos, False Neg samples within a confusions matrx (potentiall larger than 2x2)
    >>> matrix = [[5, 3, 0], [2, 3, 1], [0, 2, 11]]
    >>> columns=['Cat', 'Dog', 'Rabbit']
    >>> x = np.array([[tc, pc] for (tc, row) in enumerate(matrix) for (pc, n) in enumerate(row) for i in range(n)])
    >>> c = Confusion([(columns[i], columns[j]) for (i, j) in x], columns=['Actual', 'Predicted'])
    >>> c
    Predicted  Cat  Dog  Rabbit
    Actual
    Cat          5    3       0
    Dog          2    3       1
    Rabbit       0    2      11
    >>> dataframe_tptnfpfn(c, 'Rabbit')
    (11, 13, 1, 2)
    >>> dataframe_tptnfpfn(c.T, 'Rabbit')
    (11, 13, 2, 1)
    >>> c.mcc[2]
    0.77901...
    """
    labels = df.columns if labels is None else labels
    neg_labels = [label for label in labels if label != pos_label]
    tp = df[pos_label][pos_label]
    tn = sum(df[pred_label][true_label] for true_label in neg_labels for pred_label in neg_labels)
    fp = df[pos_label][neg_labels].sum()
    fn = sum(df[label][pos_label] for label in neg_labels)
    return tp, tn, fp, fn


class Confusion(pd.DataFrame):
    """Compute a confusion matrix from a dataframe of true and predicted classes (2 cols)
    Stats are computed as if each of the classes were considered "positive" and
    all the others were considered "negative"
    Attributes:
      stats (dict): {
        tpr: true positive rate  = num correct positive predicitons   / num positive samples
        tnr: true negative rate  = num correct negative predictions   / num negative samples
        fpr: false positive rate = num incorrect positive predictions / num negative samples
        fnr: true positive       = num incorrect positive predictions / num positive samples
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = Positive Predictive Value = tpr / (tpr + fpr)
        npv = Negative Predictive Value= tnr /