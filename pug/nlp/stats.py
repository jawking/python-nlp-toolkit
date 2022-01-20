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
        npv = Negative Predictive Value= tnr / (tnr + fnr)
        plr = Pos. Likelihood Ratio = Sens. / (1 - Spec.)
        nlr = Neg. Likelihood Ratio = (1 - Sens.) / Spec. }
    Reference:
      https://en.wikipedia.org/wiki/Confusion_matrix#Example
                Predicted
             Cat Dog Rabbit
      Actual
         Cat  5   3    0
         Dog  2   3    1
      Rabbit  0   2    11
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
    >>> (Confusion(c) == c).all().all()
    True
    >>> (Confusion(x, columns=['Actual', 'Predicted']).values == matrix).all()
    True
    >>> (Confusion(x).as_data() == x).all()
    True
    >>> c.stats_dict
    {
      "tpr": 0.625,
      "fpr": 0.105263157894737,
      "tnr": 0.736842105263158,
      "fnr": 0.75,
      "plr": 5.9375,
      "nlr": 1.01785714285714,
      "accuracy": {
        "Cat": 0.18518518518518517,
        "Dog": 0.1111111111111111,
        "Rabbit": 0.40740740740740738
      },
      "sensitivity": {
        "Cat": 0.625,
        "Dog": 0.5,
        "Rabbit": 0.84615384615384615
      },
      "specificity": {
        "Cat": 0.875,
        "Dog": 0.76190476190476186,
        "Rabbit": 0.88888888888888884
      },
      "mcc": {
        "Cat": 0.54155339089324317,
        "Dog": 0.23845524161913301,
        "Rabbit": 0.77901741435486016
      },
      "chi_squared": {
        "Cat": 7.9185620300751856,
        "Dog": 1.5352443609022557,
        "Rabbit": 16.385439560439561
      }
    }

    TODO: to_data() should output 2-column DataFrame with ["Pred", "True"] columns
          constructor should allow columns and index kwargs to override category labels when values are ints
    >>> df = pd.DataFrame(zip(list('PN'*3 + 'NNNN'), list('PNNPPNNPNPPNNN')), columns=['True', 'Pred'])
    >>> c = Confusion(df, sort=False)
    >>> c.get_false_positive(scalar=False)
    P    0.333333
    N    0.428571
    dtype: float64
    >>> c.false_positive
    0.33...
    >>> c.specificity
    0.57...
    >>> c.sensitivity
    0.66...
    >>> df = pd.DataFrame(np.matrix([[1]*10,[1]*10]).T, columns=['True', 'Pred'])
    >>> c = Confusion(df)
    >>> c
    Pred   1
    True
    1     10
    >>> c.sensitivity, c.specificity
    (inf, 1.0)

    `infer=True` assumes that the first label encounted is for the **positive** class
    `infer=False` (default) assumes the only class found is the **positive** class
    So confusion matrix (and stats like senstivity) are transposed.
    >>> c = Confusion(df, infer=True)
    >>> c
    Pred  0   1
    True
    0     0   0
    1     0  10
    >>> c.sensitivity, c.specificity
    (1.0, inf)
    >>> df = pd.DataFrame(np.matrix([[0,1,2,0,1,2,1,2,2,1],[0,1,2,1,2,0,0,1,2,0]]).T, columns=['True', 'Pred'])
    >>> Confusion(df)
    Pred  0  1  2
    True
    0     1  1  0
    1     2  1  1
    2     1  1  2
    >>> str(Confusion(df.astype(str))) == str(Confusion(df))
    True
    >>> Confusion(df).sensitivity
    0    0.50
    1    0.25
    2    0.50
    dtype: float64
    >>> df = pd.DataFrame(zip(list('ABC'*5 + 'CCC'), list('ABCBCACABCACCBCCCC')), columns=['True', 'Pred'])
    >>> c = Confusion(df)
    >>> c
    Pred  A  B  C
    True
    A     1  1  3
    B     2  2  1
    C     1  1  6
    >>> c.sensitivity
    A    0.20
    B    0.40
    C    0.75
    dtype: float64
    >>> c.get_false_positive()
    A    0.80
    B    0.60
    C    0.25
    dtype: float64
    >>> c.phi == c.mcc
    A    True
    B    True
    C    True
    dtype: bool
    >>> c.mcc
    A   -0.033150
    B    0.265197
    C    0.350000
    dtype: float64
    """

    _verbose = False
    _infer = None
    _scalar_stats = None
    _sort_classes = None
    _num_classes = None
    _num_samples = None
    _colnums = None

    def __init__(self, df, *args, **kwargs):
        # 2 x 2 matrix of ints interpretted as data rather than counts/Confusion unless its a DataFrame or Confusion type
        # if ((isinstance(df, pd.DataFrame) and
        #         (getattr(df, 'columns', range(len(df))) == getattr(df, 'index', range(len(df)))).all() and
        #         len(df) == len(iter(df).next()) and
        #         all(isinstance(value), int) for row in df for value in row) and
        #         ((len(iter(df).next()) > 2) or isinstance(df, pd.DataFrame))):
        try:
            assert(isinstance(df, pd.DataFrame))
            assert((df.columns == df.index).all())
            assert(df.values.dtype == int)
            self.construct_copy(df, *args, **kwargs)
            return
        except (AssertionError, AttributeError, ValueError):
            pass
        _infer = kwargs.pop('infer', kwargs.pop('infer_classes', None))
        _sort_classes = kwargs.pop('sort', kwargs.pop('sort_classes', 1))
        verbose = kwargs.pop('verbose', False)

        df = make_dataframe(df, *args, **kwargs)
        # only 2 columns allowed: true class and predicted class
        columns = df.columns[:2]
        # to maintain the order of classes:
        # pd.Index(df[df.columns[0]]).intersection(pd.Index(df[df.columns[1]])).unique
        # or pip install orderedset by Raymond Hettinger and Rob Speer
        index = pd.Index(np.concatenate([df[columns[0]], df[columns[1]]])).unique()

        if _infer and len(index) == 1:
            index = np.concatenate([index, [infer_pos_label(index[0])]])
            # if verbose:
            #     print('WARN: Only a single class label was found} so inferring (guessing) a positive label.')
        index = pd.Index(pd.Index(index).unique())
        if _sort_classes:
            index = index.sort_values(ascending=(_sort_classes > 0))

        # construct an empty parent DataFrame instance
        super(Confusion, self).__init__(index=pd.Index(index, name=columns[0]), columns=pd.Index(index, name=columns[1]))

        # metadata to speed other operations
        self._verbose = verbose
        self._infer = _infer
        self._scalar_stats = kwargs.pop('scalar', kwargs.pop('scalar_stats', None))
        self._sort_classes = _sort_classes
        self._num_classes = len(index)
        self._num_samples = len(df)
        self._colnums = np.arange(0, self._num_classes)
        # look for Positive and Negative column labels by first finding columns labeled
        #    "Negative", "-1", "0", "Healthy", "N/A", etc

        try:
            self._neg_label = (label for label in self.columns if unicode(label).strip().lower()[0] in ('-nh0')).next()
        except StopIteration:
            self._neg_label = self.columns[-1]
        try:
            self._pos_label = (label for label in self.columns if label != self._neg_label).next()
        except StopIteration:
            self._pos_label = infer_pos_label(self._neg_label)

        logger.debug('true class samples: {}'.format(df[columns[0]].values[:5]))
        for p_class in index:
            self[p_class] = pd.Series([len(df[(df[columns[0]] == t_class) & (df[columns[1]] == p_class)]) for t_class in index],
                                      index=index, dtype=int)
        self.refresh_meta()

    def construct_copy(self, other, *args, **kwargs):
        # construct a parent DataFrame instance
        parent_type = super(Confusion, self)
        parent_type.__init__(other)
        try:
            for k, v in other.__dict__.iteritems():
                if hasattr(parent_type, k) and hasattr(self, k) and getattr(parent_type, k) == getattr(self, k):
                    continue
                setattr(self, k, deepcopy(v))
        except AttributeError:
            pass
        self.refresh_meta()

    def as_data(self):
        return np.array([[tc, pc] for (tc, row) in enumerate(self.values) for (pc, n) in enumerate(row) for i in range(n)])

    def refresh_meta(self):
        """Calculations that only depend on aggregate counts in Confusion Matrix go here"""

        # these calcs are duplicated in __init__()
        self._num_classes = len(self.index)
        self._colnums = np.arange(0, self._num_classes)
        try:
            self._neg_label = (label for label in self.columns if unicode(label).strip().lower()[0] in ('-nh0')).next()
        except StopIteration:
            self._neg_label = self.columns[-1]
        try:
            self._pos_label = (label for label in self.columns if label != self._neg_label).next()
        except StopIteration:
            self._pos_label = infer_pos_label(self._neg_label)

        # TODO: reorder columns with newly guessed pos and neg class labels first

        # TODO: gather up additional meta calculations here so
        #       a Confusion matrix can be build from an existing DataFrame that contains confusion counts
        #       rather than just two columns of labels.
        self._hist_labels = self.sum().astype(int)
        self._num_total = self._hist_labels.sum()
        assert(self._num_total == self.sum().sum())
        self._num_pos_labels = self._hist_labels.get(self._pos_label, 0)
        self._num_neg_labels = self._num_total - self._num_pos_labels  # everything that isn't positive is negative
        self._hist_classes = self.T.sum()
