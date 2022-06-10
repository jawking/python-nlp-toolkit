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

   