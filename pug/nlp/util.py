#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Utilities for Natural Language Processing (NLP):

* Vocabulary and dimension reduction
* Word statistics calculation
* Add a timezone to a datetime
* Slice a django queryset
* Genereate batches from a long list or sequence
* Inverse dict/hashtable lookup
* Generate a valid python variable or class name from a string
* Generate a slidedeck-compatible markdown from an text or markdown outline or list
* Convert a sequence of sequences to a dictionary of sequences
* Pierson correlation coefficient calculation
* Parse a string into sentences or tokens
* Table (list of list) manipulation
* make_time, make_date, quantize_datetime -- ignore portions of a datetime struct
* ordinal_float, datetime_from_ordinal_float -- conversion between datetimes and float days
* days_since    -- subract two date or datetime objects and return difference in days (float)

'''
from __future__ import division, print_function, absolute_import  # , unicode_literals
# from future import standard_library
# standard_library.install_aliases()  # noqa
from builtins import next
from builtins import map
from builtins import zip
from builtins import chr
from builtins import range
from builtins import object
from builtins import str  # noqa
from future.utils import viewitems
from past.builtins import basestring
try:  # python 3.5+
    from io import StringIO
    # from ConfigParser import ConfigParser

except:
    from io import StringIO
    # from configparser import ConfigParser

import os
import itertools
import datetime
import types
import re
import string
import csv
import logging
import warnings
from traceback import print_exc
from collections import OrderedDict, Mapping, Counter
from itertools import islice
from decimal import Decimal, InvalidOperation, InvalidContext
import math
import copy
import codecs
import json
from threading import _get_ident
from time import mktime
from traceback import format_exc

import pandas as pd
from .tutil import clip_datetime
import progressbar
from fuzzywuzzy import process as fuzzy
from slugify import slugify

from pug.nlp import charlist

from .constant import PUNC
from .constant import FLOAT_TYPES, MAX_CHR
from .constant import ROUNDABLE_NUMERIC_TYPES, COUNT_NAMES, SCALAR_TYPES, NUMBERS_AND_DATETIMES
from .constant import DATETIME_TYPES, DEFAULT_TZ

from pug.nlp import regex as rex
from .tutil import make_tz_aware


np = pd.np
logger = logging.getLogger(__name__)


def qs_to_table(qs, excluded_fields=['id']):
    rows, rowl = [], []
    qs = qs.all()
    fields = sorted(qs[0]._meta.get_all_field_names())
    for row in qs:
        for f in fields:
            if f in excluded_fields:
                continue
            rowl += [getattr(row, f)]
        rows, rowl = rows + [rowl], []
    return rows


def force_hashable(obj, recursive=True):
    """Force frozenset() command to freeze the order and contents of mutables and iterables like lists, dicts, generators

    Useful for memoization and constructing dicts or hashtables where keys must be immutable.

    FIXME: Rename function because "hashable" is misleading.
           A better name might be `force_immutable`.
           because some hashable objects (generators) are tuplized  by this function
           `tuplized` is probably a better name, but strings are left alone, so not quite right

    >>> force_hashable([1,2.,['3','four'],'five', {'s': 'ix'}])
    (1, 2.0, ('3', 'four'), 'five', (('s', 'ix'),))
    >>> force_hashable(i for i in range(4))
    (0, 1, 2, 3)
    >>> force_hashable(Counter('abbccc')) ==  (('a', 1), ('c', 3), ('b', 2))
    True
    """
    # if it's already hashable, and isn't a generator (which are also hashable, but not mutable)
    if hasattr(obj, '__hash__') and not hasattr(obj, 'next'):
        try:
            hash(obj)
            return obj
        except:
            pass
    if hasattr(obj, '__iter__'):
        # looks like a Mapping if it has .get() and .items(), so should treat it like one
        if hasattr(obj, 'get') and hasattr(obj, 'items'):
            # FIXME: prevent infinite recursion:
            #        tuples don't have 'items' method so this will recurse forever
            #        if elements within new tuple aren't hashable and recurse has not been set!
            return force_hashable(tuple(obj.items()))
        if recursive:
            return tuple(force_hashable(item) for item in obj)
        return tuple(obj)
    # strings are hashable so this ends the recursion for any object without an __iter__ method (strings do not)
    return str(obj)


def inverted_dict(d):
    """Return a dict with swapped keys and values

    >>> inverted_dict({0: ('a', 'b'), 1: 'cd'}) == {'cd': 1, ('a', 'b'): 0}
    True
    """
    return dict((force_hashable(v), k) for (k, v) in viewitems(dict(d)))


def inverted_dict_of_lists(d):
    """Return a dict where the keys are all the values listed in the values of the original dict

    >>> inverted_dict_of_lists({0: ['a', 'b'], 1: 'cd'}) == {'a': 0, 'b': 0, 'cd': 1}
    True
    """
    new_dict = {}
    for (old_key, old_value_list) in viewitems(dict(d)):
        for new_key in listify(old_value_list):
            new_dict[new_key] = old_key
    return new_dict


def sort_strings(strings, sort_order=None, reverse=False, case_sensitive=False, sort_order_first=True):
    """Sort a list of strings according to the provided sorted list of string prefixes

    TODO:
        - Provide an option to use `.startswith()` rather than a fixed prefix length (will be much slower)

    Arguments:
        sort_order_first (bool): Whether strings in sort_order should always preceed "unknown" strings
        sort_order (sequence of str): Desired ordering as a list of prefixes to the strings
            If sort_order strings have varying length, the max length will determine the prefix length compared
        reverse (bool): whether to reverse the sort orded. Passed through to `sorted(strings, reverse=reverse)`
        case_senstive (bool): Whether to sort in lexographic rather than alphabetic order
         and whether the prefixes  in sort_order are checked in a case-sensitive way

    Examples:
        >>> sort_strings(['morn32', 'morning', 'unknown', 'date', 'dow', 'doy', 'moy'],
        ...              ('dat', 'dow', 'moy', 'dom', 'doy', 'mor'))
        ['date', 'dow', 'moy', 'doy', 'morn32', 'morning', 'unknown']
        >>> sort_strings(['morn32', 'morning', 'unknown', 'less unknown', 'lucy', 'date', 'dow', 'doy', 'moy'],
        ...              ('dat', 'dow', 'moy', 'dom', 'doy', 'mor'), reverse=True)
        ['unknown', 'lucy', 'less unknown', 'morning', 'morn32', 'doy', 'moy', 'dow', 'date']

        Strings whose prefixes don't exist in `sort_order` sequence can be interleaved into the
        sorted list in lexical order by setting `sort_order_first=False`
        >>> sort_strings(['morn32', 'morning', 'unknown', 'lucy', 'less unknown', 'date', 'dow', 'doy', 'moy'],
        ...              ('dat', 'dow', 'moy', 'dom', 'moy', 'mor'),
        ...              sort_order_first=False)  # doctest: +NORMALIZE_WHITESPACE
        ['date', 'dow', 'doy', 'less unknown', 'lucy', 'moy', 'morn32', 'morning', 'unknown']
    """
    if not case_sensitive:
        sort_order = tuple(s.lower() for s in sort_order)
        strings = tuple(s.lower() for s in strings)
    prefix_len = max(l