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
    prefix_len = max(len(s) for s in sort_order)

    def compare(a, b, prefix_len=prefix_len):
        if prefix_len:
            if a[:prefix_len] in sort_order:
                if b[:prefix_len] in sort_order:
                    comparison = sort_order.index(a[:prefix_len]) - sort_order.index(b[:prefix_len])
                    comparison = int(comparison / abs(comparison or 1))
                    if comparison:
                        return comparison * (-2 * reverse + 1)
                elif sort_order_first:
                    return -1 * (-2 * reverse + 1)
            # b may be in sort_order list, so it should be first
            elif sort_order_first and b[:prefix_len] in sort_order:
                return -2 * reverse + 1
        return (-1 * (a < b) + 1 * (a > b)) * (-2 * reverse + 1)

    return sorted(strings, cmp=compare)


def clean_field_dict(field_dict, cleaner=str.strip, time_zone=None):
    r"""Normalize field values by stripping whitespace from strings, localizing datetimes to a timezone, etc

    >>> sorted(clean_field_dict({'_state': object(), 'x': 1, 'y': u"\t  Wash Me! \n" }).items())
    [('x', 1), ('y', u'Wash Me!')]
    """
    d = {}
    if time_zone is None:
        tz = DEFAULT_TZ
    for k, v in viewitems(field_dict):
        if k == '_state':
            continue
        if isinstance(v, basestring):
            d[k] = cleaner(str(v))
        elif isinstance(v, (datetime.datetime, datetime.date)):
            d[k] = tz.localize(v)
        else:
            d[k] = v
    return d


# def reduce_vocab(tokens, similarity=.85, limit=20):
#     """Find spelling variations of similar words within a list of tokens to reduce token set size

#     Arguments:
#       tokens (list or set or tuple of str): token strings from which to eliminate similar spellings

#     Examples:
#       >>> reduce_vocab(('on', 'hon', 'honey', 'ones', 'one', 'two', 'three'))  # doctest: +NORMALIZE_WHITESPACE


#     """
#     tokens = set(tokens)
#     thesaurus = {}
#     while tokens:
#         tok = tokens.pop()
#         matches = fuzzy.extractBests(tok, tokens, score_cutoff=int(similarity * 100), limit=20)
#         if matches:
#             thesaurus[tok] = zip(*matches)[0]
#         else:
#             thesaurus[tok] = (tok,)
#         for syn in thesaurus[tok][1:]:
#             tokens.discard(syn)
#     return thesaurus


def reduce_vocab(tokens, similarity=.85, limit=20, sort_order=-1):
    """Find spelling variations of similar words within a list of tokens to reduce token set size

    Lexically sorted in reverse order (unless `reverse=False`), before running through fuzzy-wuzzy
    which results in the longer of identical spellings to be prefered (e.g. "ones" prefered to "one")
    as the key token. Usually you wantThis is usually what you want.

    Arguments:
      tokens (list or set or tuple of str): token strings from which to eliminate similar spellings
      similarity (float): portion of characters that should be unchanged in order to be considered a synonym
        as a fraction of the key token length.
        e.g. `0.85` (which means 85%) allows "hon" to match "on" and "honey", but not "one"

    Returns:
      dict: { 'token': ('similar_token', 'similar_token2', ...), ...}

    Examples:
      >>> tokens = ('on', 'hon', 'honey', 'ones', 'one', 'two', 'three')
      >>> answer = {'hon': ('on', 'honey'),
      ...           'one': ('ones',),
      ...           'three': (),
      ...           'two': ()}
      >>> reduce_vocab(tokens, sort_order=1) == answer
      True
      >>> answer = {'honey': ('hon',),
      ...           'ones': ('on', 'one'),
      ...           'three': (),
      ...           'two': ()}
      >>> reduce_vocab(tokens, sort_order=-1) == answer
      True
      >>> (reduce_vocab(tokens, similarity=0.3, limit=2, sort_order=-1) ==
      ...  {'ones': (), 'two': ('on', 'hon'), 'three': ('honey', 'one')})
      True
      >>> (reduce_vocab(tokens, similarity=0.3, limit=3, sort_order=-1) ==
      ...  {'ones': (), 'two': ('on', 'hon', 'one'), 'three': ('honey',)})
      True
    """
    if 0 <= similarity <= 1:
        similarity *= 100
    if sort_order:
        tokens = set(tokens)
        tokens_sorted = sorted(list(tokens), reverse=bool(sort_order < 0))
    else:
        tokens_sorted = list(tokens)
        tokens = set(tokens)
    # print(tokens)
    thesaurus = {}
    for tok in tokens_sorted:
        try:
            tokens.remove(tok)
        except (KeyError, ValueError):
            continue
        # FIXME: this is slow because the tokens list must be regenerated and reinstantiated with each iteration
        matches = fuzzy.extractBests(tok, list(tokens), score_cutoff=int(similarity), limit=limit)
        if matches:
            thesaurus[tok] = list(zip(*matches))[0]
        else:
            thesaurus[tok] = ()
        for syn in thesaurus[tok]:
            tokens.discard(syn)
    return thesaurus


def reduce_vocab_by_len(tokens, similarity=.87, limit=20, reverse=True):
    """Find spelling variations of similar words within a list of tokens to reduce token set size

    Sorted by length (longest first unless reverse=False) before running through fuzzy-wuzzy
    which results in longer key tokens.

    Arguments:
      tokens (list or set or tuple of str): token strings from which to eliminate similar spellings

    Returns:
      dict: { 'token': ('similar_token', 'similar_token2', ...), ...}

    Examples:
      >>> tokens = ('on', 'hon', 'honey', 'ones', 'one', 'two', 'three')
      >>> reduce_vocab_by_len(tokens) ==  {'honey': ('on', 'hon', 'one'), 'ones': (), 'three': (), 'two': ()}
      True
    """
    tokens = set(tokens)
    tokens_sorted = list(zip(*sorted([(len(tok), tok) for tok in tokens], reverse=reverse)))[1]
    return reduce_vocab(tokens=tokens_sorted, similarity=similarity, limit=limit, sort_order=0)


def quantify_field_dict(field_dict, precision=None, date_precision=None, cleaner=str.strip):
    r"""Convert strings and datetime objects in the values of a dict into float/int/long, if possible

    Arguments:
      field_dict (dict): The dict to have any values (not keys) that are strings "quantified"
      precision (int): Number of digits of precision to enforce
      cleaner: A string cleaner to apply to all string before


    FIXME: define a time zone for the datetime object and get it to be consistent for travis and local

    >>> sorted(viewitems(quantify_field_dict({'_state': object(), 'x': 12345678911131517L, 'y': "\t  Wash Me! \n",
    ...     'z': datetime.datetime(1970, 10, 23, 23, 59, 59, 123456)})))  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    [('x', 12345678911131517), ('y', u'Wash Me!'), ('z', 25...99.123456)]
    """
    if cleaner:
        d = clean_field_dict(field_dict, cleaner=cleaner)
    for k, v in viewitems(d):
        if isinstance(d[k], datetime.datetime):
            # seconds since epoch = datetime.datetime(1969,12,31,18,0,0)
            try:
                # around the year 2250, a float conversion of this string will lose 1 microsecond of precision,
                # and around 22500 the loss of precision will be 10 microseconds
                d[k] = float(d[k].strftime('%s.%f'))  # seconds since Jan 1, 1970
                if date_precision is not None and isinstance(d[k], ROUNDABLE_NUMERIC_TYPES):
                    d[k] = round(d[k], date_precision)
                    continue
            except:
                pass
        if not isinstance(d[k], (int, float)):
            try:
                d[k] = float(d[k])
            except:
                pass
        if precision is not None and isinstance(d[k], ROUNDABLE_NUMERIC_TYPES):
            d[k] = round(d[k], precision)
        if isinstance(d[k], float) and d[k].is_integer():
            # `int()` will convert to a long, if value overflows an integer type
            # use the original value, `v`, in case it was a long and d[k] is has been truncated by the conversion to float!
            d[k] = int(v)
    return d


def generate_batches(sequence, batch_len=1, allow_partial=True, ignore_errors=True, verbosity=1):
    """Iterate through a sequence (or generator) in batches of length `batch_len`

    http://stackoverflow.com/a/761125/623735
    >>> [batch for batch in generate_batches(range(7), 3)]
    [[0, 1, 2], [3, 4, 5], [6]]
    """
    it = iter(sequence)
    last_value = False
    # An exception will be thrown by `.next()` here and caught in the loop that called this iterator/generator
    while not last_value:
        batch = []
        for n in range(batch_len):
            try:
                batch += (next(it),)
            except StopIteration:
                last_value = True
                if batch:
                    break
                else:
                    raise StopIteration
            except Exception:
                # 'Error: new-line character seen in unquoted field - do you need to open the file in universal-newline mode?'
                if verbosity > 0:
                    print_exc()
                if not ignore_errors:
                    raise
        yield batch


def generate_tuple_batches(qs, batch_len=1):
    """Iterate through a queryset in batches of length `batch_len`

    >>> [batch for batch in generate_tuple_batches(range(7), 3)]
    [(0, 1, 2), (3, 4, 5), (6,)]
    """
    num_items, batch = 0, []
    for item in qs:
        if num_items >= batch_len:
            yield tuple(batch)
            num_items = 0
            batch = []
        num_items += 1
        batch += [item]
    if num_items:
        yield tuple(batch)


def sliding_window(seq, n=2):
    """Generate overlapping sliding/rolling windows (of width n) over an iterable

    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

    References:
      http://stackoverflow.com/a/6822773/623735

    Examples:

    >>> list(sliding_window(range(6), 3))  # doctest: +NORMALIZE_WHITESPACE
    [(0, 1, 2),
     (1, 2, 3),
     (2, 3, 4),
     (3, 4, 5)]
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def generate_slices(sliceable_set, batch_len=1, length=None, start_batch=0):
    """Iterate through a sequence (or generator) in batches of length `batch_len`

    See Also:
      pug.dj.db.generate_queryset_batches

    References:
      http://stackoverflow.com/a/761125/623735

    Examples:
      >>  [batch for batch in generate_slices(range(7), 3)]
      [(0, 1, 2), (3, 4, 5), (6,)]
      >>  from django.contrib.auth.models import User, Permission
      >>  len(list(generate_slices(User.objects.all(), 2)))       == max(math.ceil(User.objects.count() / 2.), 1)
      True
      >>  len(list(generate_slices(Permission.objects.all(), 2))) == max(math.ceil(Permission.objects.count() / 2.), 1)
      True
    """
    if length is None:
        try:
            length = sliceable_set.count()
        except:
            length = len(sliceable_set)
    length = int(length)

    for i in range(int(length / batch_len + 1)):
        if i < start_batch:
            continue
        start = i * batch_len
        end = min((i + 1) * batch_len, length)
        if start != end:
            yield tuple(sliceable_set[start:end])
    raise StopIteration


def find_count_label(d):
    """Find the member of a set that means "count" or "frequency" or "probability" or "number of occurrences".

    """
    for name in COUNT_NAMES:
        if name in d:
            return name
    for name in COUNT_NAMES:
        if str(name).lower() in d:
            return name


def first_in_seq(seq):
    # lists/sequences
    return next(iter(seq))


def get_key_for_value(dict_obj, value, default=None):
    """
    >>> get_key_for_value({0: 'what', 'k': 'ever', 'you': 'want', 'to find': None}, 'you')
    >>> get_key_for_value({0: 'what', 'k': 'ever', 'you': 'want', 'to find': None}, 'you', default='Not Found')
    'Not Found'
    >>> get_key_for_value({0: 'what', 'k': 'ever', 'you': 'want', 'to find': None}, 'other', default='Not Found')
    'Not Found'
    >>> get_key_for_value({0: 'what', 'k': 'ever', 'you': 'want', 'to find': None}, 'want')
    'you'
    >>> get_key_for_value({0: 'what', '': 'ever', 'you': 'want', 'to find': None, 'you': 'too'}, 'what')
    0
    >>> get_key_for_value({0: 'what', '': 'ever', 'you': 'want', 'to find': None, 'you': 'too', ' ': 'want'}, 'want')
    ' '
    """
    for k, v in viewitems(dict_obj):
        if v == value:
            return k
    return default


def list_set(seq):
    """Similar to `list(set(seq))`, but maintains the order of `seq` while eliminating duplicates

    In general list(set(L)) will not have the same order as the original list.
    This is a list(set(L)) function that will preserve the order of L.

    Arguments:
      seq (iterable): list, tuple, or other iterable to be used to produce an ordered `set()`

    Returns:
      iterable: A copy of `seq` but with duplicates removed, and distinct elements in the same order as in `seq`

    Examples:
      >>> list_set([2.7,3,2,2,2,1,1,2,3,4,3,2,42,1])
      [2.7, 3, 2, 1, 4, 42]
      >>> list_set(['Zzz','abc', ('what.', 'ever.'), 0, 0.0, 'Zzz', 0.00, 'ABC'])
      ['Zzz', 'abc', ('what.', 'ever.'), 0, 'ABC']
    """
    new_list = []
    for i in seq:
        if i not in new_list:
            new_list += [i]
    return type(seq)(new_list)


def fuzzy_get(possible_keys, approximate_key, default=None, similarity=0.6, tuple_joiner='|', key_and_value=False, dict_keys=None):
    r"""Find the closest matching key in a dictionary (or element in a list)

    For a dict, optionally retrieve the associated value associated with the closest key

    Notes:
      `possible_keys` must have all string elements or keys!
      Argument order is in reverse order relative to `fuzzywuzzy.process.extractOne()`
        but in the same order as get(self, key) method on dicts

    Arguments:
      possible_keys (dict): object to run the get method on using the key that is most similar to one within the dict
      approximate_key (str): key to look for a fuzzy match within the dict keys
      default (obj): the value to return if a similar key cannote be found in the `possible_keys`
      similarity (float): fractional similiarity between the approximate_key and the dict key (0.9 means 90% of characters must be identical)
      tuple_joiner (str): Character to use as delimitter/joiner between tuple elements.
        Used to create keys of any tuples to be able to use fuzzywuzzy string matching on it.
      key_and_value (bool): Whether to return both the key and its value (True) or just the value (False).
        Default is the same behavior as dict.get (i.e. key_and_value=False)
      dict_keys (list of str): if you already have a set of keys to search, this will save this funciton a little time and RAM

    See Also:
      get_similar: Allows nonstring keys and searches object attributes in addition to keys

    Examples:
      >>> fuzzy_get({'seller': 2.7, 'sailor': set('e')}, 'sail')
      set(['e'])
      >>> fuzzy_get({'seller': 2.7, 'sailor': set('e'), 'camera': object()}, 'SLR')
      2.7
      >>> fuzzy_get({'seller': 2.7, 'sailor': set('e'), 'camera': object()}, 'I')
      set(['e'])
      >>> fuzzy_get({'word': tuple('word'), 'noun': tuple('noun')}, 'woh!', similarity=.3, key_and_value=True)
      ('word', ('w', 'o', 'r', 'd'))
      >>> fuzzy_get({'word': tuple('word'), 'noun': tuple('noun')}, 'woh!', similarity=.9, key_and_value=True)
      (None, None)
      >>> fuzzy_get({'word': tuple('word'), 'noun': tuple('noun')}, 'woh!', similarity=.9, default='darn :-()', key_and_value=True)
      (None, 'darn :-()')
      >>> possible_keys = ('alerts astronomy conditions currenthurricane forecast forecast10day geolookup history ' +
      ...                  'hourly hourly10day planner rawtide satellite tide webcams yesterday').split()
      >>> fuzzy_get(possible_keys, "cond")
      'conditions'
      >>> fuzzy_get(possible_keys, "Tron")
      'astronomy'
      >>> df = pd.DataFrame(np.arange(6*2).reshape(2,6), columns=('alpha','beta','omega','begin','life','end'))
      >>> fuzzy_get(df, 'beg')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
      0    3
      1    9
      Name: begin, dtype: int...
      >>> fuzzy_get(df, 'get')
      >>> fuzzy_get(df, 'et')[1]
      7
      >>> fuzzy_get(df, 'get')
    """
    dict_obj = copy.copy(possible_keys)
    if not isinstance(dict_obj, (Mapping, pd.DataFrame, pd.Series)):
        dict_obj = dict((x, x) for x in dict_obj)

    fuzzy_key, value = None, default
    if approximate_key in dict_obj:
        fuzzy_key, value = approximate_key, dict_obj[approximate_key]
    else:
        strkey = str(approximate_key)
        if approximate_key and strkey and strkey.strip():
            # print 'no exact match was found for {0} in {1} so preprocessing keys'.format(approximate_key, dict_obj.keys())
            if any(isinstance(k, (tuple, list)) for k in dict_obj):
                dict_obj = dict((tuple_joiner.join(str(k2) for k2 in k), v) for (k, v) in viewitems(dict_obj))
                if isinstance(approximate_key, (tuple, list)):
                    strkey = tuple_joiner.join(approximate_key)
            # fuzzywuzzy requires that dict_keys be a list (sets and tuples fail!)
            dict_keys = list(set(dict_keys if dict_keys else dict_obj))
            if strkey in dict_keys:
                fuzzy_key, value = strkey, dict_obj[strkey]
            else:
                strkey = strkey.strip()
                if strkey in dict_keys:
                    fuzzy_key, value = strkey, dict_obj[strkey]
                else:
                    fuzzy_key_scores = fuzzy.extractBests(strkey, dict_keys, score_cutoff=min(max(similarity * 100.0 - 1, 0), 100), limit=6)
                    if fuzzy_key_scores:
                        fuzzy_score_keys = []
                        # add length similarity as part of score
                        for (i, (k, score)) in enumerate(fuzzy_key_scores):
                            fuzzy_score_keys += [(score * math.sqrt(len(strkey)**2 / float((len(k)**2 + len(strkey)**2) or 1)), k)]
                        fuzzy_score, fuzzy_key = sorted(fuzzy_score_keys)[-1]
                        value = dict_obj[fuzzy_key]
    if key_and_value:
        if key_and_value in ('v', 'V', 'value', 'VALUE', 'Value'):
            return value
        return fuzzy_key, value
    else:
        return value


def fuzzy_get_value(obj, approximate_key, default=None, **kwargs):
    """ Like fuzzy_get, but assume the obj is dict-like and return the value without the key

    Notes:
      Argument order is in reverse order relative to `fuzzywuzzy.process.extractOne()`
        but in the same order as get(self, key) method on dicts

    Arguments:
      obj (dict-like): object to run the get method on using the key that is most similar to one within the dict
      approximate_key (str): key to look for a fuzzy match within the dict keys
      default (obj): the value to return if a similar key cannote be found in the `possible_keys`
      similarity (str): fractional similiarity between the approximate_key and the dict key (0.9 means 90% of characters must be identical)
      tuple_joiner (str): Character to use as delimitter/joiner between tuple elements.
        Used to create keys of any tuples to be able to use fuzzywuzzy string matching on it.
      key_and_value (bool): Whether to return both the key and its value (True) or just the value (False).
        Default is the same behavior as dict.get (i.e. key_and_value=False)
      dict_keys (list of str): if you already have a set of keys to search, this will save this funciton a little time and RAM

    Examples:
      >>> fuzzy_get_value({'seller': 2.7, 'sailor': set('e')}, 'sail') == set(['e'])
      True
      >>> fuzzy_get_value({'seller': 2.7, 'sailor': set('e'), 'camera': object()}, 'SLR')
      2.7
      >>> fuzzy_get_value({'seller': 2.7, 'sailor': set('e'), 'camera': object()}, 'I') == set(['e'])
      True
      >>> fuzzy_get_value({'word': tuple('word'), 'noun': tuple('noun')}, 'woh!', similarity=.3)
      ('w', 'o', 'r', 'd')
      >>> df = pd.DataFrame(np.arange(6*2).reshape(2,6), columns=('alpha','beta','omega','begin','life','end'))
      >>> fuzzy_get_value(df, 'life')[0], fuzzy_get(df, 'omega')[0]
      (4, 2)
    """
    dict_obj = OrderedDict(obj)
    try:
        return dict_obj[list(dict_obj.keys())[int(approximate_key)]]
    except (ValueError, IndexError):
        pass
    return fuzzy_get(dict_obj, approximate_key, key_and_value=False, **kwargs)


def fuzzy_get_tuple(dict_obj, approximate_key, dict_keys=None, key_and_value=False, similarity=0.6, default=None):
    """Find the closest matching key and/or value in a dictionary (must have all string keys!)"""
    return fuzzy_get(dict(('|'.join(str(k2) for k2 in k), v) for (k, v) in viewitems(dict_obj)),
                     '|'.join(str(k) for k in approximate_key), dict_keys=dict_keys,
                     key_and_value=key_and_value, similarity=similarity, default=default)


def sod_transposed(seq_of_dicts, align=True, pad=True, filler=None):
    """Return sequence (list) of dictionaries, transposed into a dictionary of sequences (lists)

    >>> sorted(sod_transposed([{'c': 1, 'cm': u'P'}, {'c': 1, 'ct': 2, 'cm': 6, 'cn': u'MUS'}, {'c': 1, 'cm': u'Q', 'cn': u'ROM'}], filler=0).items())
    [('c', [1, 1, 1]), ('cm', [u'P', 6, u'Q']), ('cn', [0, u'MUS', u'ROM']), ('ct', [0, 2, 0])]
    >>> sorted(sod_transposed(({'c': 1, 'cm': u'P'}, {'c': 1, 'ct': 2, 'cm': 6, 'cn': u'MUS'}, {'c': 1, 'cm': u'Q', 'cn': u'ROM'}),
    ...                       filler=0, align=0).items())
    [('c', [1, 1, 1]), ('cm', [u'P', 6, u'Q']), ('cn', [u'MUS', u'ROM']), ('ct', [2])]
    """
    result = {}
    if isinstance(seq_of_dicts, Mapping):
        seq_of_dicts = [seq_of_dicts]
    it = iter(seq_of_dicts)
    # if you don't need to align and/or fill, then just loop through and return
    if not (align and pad):
        for d in it:
            for k in d:
                result[k] = result.get(k, []) + [d[k]]
        return result
    # need to align and/or fill, so pad as necessary
    for i, d in enumerate(it):
        for k in d:
            result[k] = result.get(k, [filler] * (i * int(align))) + [d[k]]
        for k in result:
            if k not in d:
                result[k] += [filler]
    return result


def joined_seq(seq, sep=None):
    r"""Join a sequence into a tuple or a concatenated string

    >>> joined_seq(range(3), ', ')
    u'0, 1, 2'
    >>> joined_seq([1, 2, 3])
    (1, 2, 3)
    """
    joined_seq = tuple(seq)
    if isinstance(sep, basestring):
        joined_seq = sep.join(str(item) for item in joined_seq)
    return joined_seq


def consolidate_stats(dict_of_seqs, stats_key=None, sep=','):
    """Join (stringify and concatenate) keys (table fields) in a dict (table) of sequences (columns)

    >>> consolidate_stats(dict([('c', [1, 1, 1]), ('cm', [u'P', 6, u'Q']), ('cn', [0, u'MUS', u'ROM']), ('ct', [0, 2, 0])]), stats_key='c')
    [{u'P,0,0': 1}, {u'6,MUS,2': 1}, {u'Q,ROM,0': 1}]
    >>> consolidate_stats([{'c': 1, 'cm': 'P', 'cn': 0, 'ct': 0}, {'c': 1, 'cm': 6, 'cn': 'MUS', 'ct': 2},
    ...                    {'c': 1, 'cm': 'Q', 'cn': 'ROM', 'ct': 0}], stats_key='c')
    [{u'P,0,0': 1}, {u'6,MUS,2': 1}, {u'Q,ROM,0': 1}]
    """
    if isinstance(dict_of_seqs, dict):
        stats = dict_of_seqs[stats_key]
        keys = joined_seq(sorted([k for k in dict_of_seqs if k is not stats_key]), sep=None)
        joined_key = joined_seq(keys, sep=sep)
        result = {stats_key: [], joined_key: []}
        for i, statistic in enumerate(stats):
            result[stats_key] += [statistic]
            result[joined_key] += [joined_seq((dict_of_seqs[k][i] for k in keys if k is not stats_key), sep)]
        return list({k: result[stats_key][i]} for i, k in enumerate(result[joined_key]))
    return [{joined_seq((d[k] for k in sorted(d) if k is not stats_key), sep): d[stats_key]} for d in dict_of_seqs]


def dos_from_table(table, header=None):
    """Produce dictionary of sequences from sequence of sequences, optionally with a header "row".

    >>> dos_from_table([['hello', 'world'], [1, 2], [3,4]]) == {'hello': [1, 3], 'world': [2, 4]}
    True
    """
    start_row = 0
    if not table:
        return table
    if not header:
        header = table[0]
        start_row = 1
    header_list = header
    if header and isinstance(header, basestring):
        header_list = header.split('\t')
        if len(header_list) != len(table[0]):
            header_list = header.split(',')
        if len(header_list) != len(table[0]):
            header_list = header.split(' ')
    ans = {}
    for i, k in enumerate(header):
        ans[k] = [row[i] for row in table[start_row:]]
    return ans


def transposed_lists(list_of_lists, default=None):
    """Like `numpy.transposed`, but allows uneven row lengths

    Uneven lengths will affect the order of the elements in the rows of the transposed lists

    >>> transposed_lists([[1, 2], [3, 4, 5], [6]])
    [[1, 3, 6], [2, 4], [5]]
    >>> transposed_lists(transposed_lists([[], [1, 2, 3], [4]]))
    [[1, 2, 3], [4]]
    >>> l = transposed_lists([range(4),[4,5]])
    >>> l
    [[0, 4], [1, 5], [2], [3]]
    >>> transposed_lists(l)
    [[0, 1, 2, 3], [4, 5]]
    """
    if default is None or default is [] or default is tuple():
        default = []
    elif default is 'None':
        default = [None]
    else:
        default = [default]

    N = len(list_of_lists)
    Ms = [len(row) for row in list_of_lists]
    M = max(Ms)
    ans = []
    for j in range(M):
        ans += [[]]
        for i in range(N):
            if j < Ms[i]:
                ans[-1] += [list_of_lists[i][j]]
            else:
                ans[-1] += list(default)
    return ans


def transposed_matrix(matrix, filler=None, row_type=list, matrix_type=list, value_type=None):
    """Like numpy.transposed, evens up row (list) lengths that aren't uniform, filling with None.

    Also, makes all elements a uniform type (default=type(matrix[0][0])),
    except for filler elements.

    TODO: add feature to delete None's at the end of rows so that transpose(transpose(LOL)) = LOL

    >>> transposed_matrix([[1, 2], [3, 4, 5], [6]])
    [[1, 3, 6], [2, 4, None], [None, 5, None]]
    >>> transposed_matrix(transposed_matrix([[1, 2], [3, 4, 5], [6]]))
    [[1, 2, None], [3, 4, 5], [6, None, None]]
    >>> transposed_matrix([[], [1, 2, 3], [4]])  # empty first row forces default value type (float)
    [[None, 1.0, 4.0], [None, 2.0, None], [None, 3.0, None]]
    >>> transposed_matrix(transposed_matrix([[], [1, 2, 3], [4]]))
    [[None, None, None], [1.0, 2.0, 3.0], [4.0, None, None]]
    >>> l = transposed_matrix([range(4),[4,5]])
    >>> l
    [[0, 4], [1, 5], [2, None], [3, None]]
    >>> transposed_matrix(l)
    [[0, 1, 2, 3], [4, 5, None, None]]
    >>> transposed_matrix([[1,2],[1],[1,2,3]])
    [[1, 1, 1], [2, None, 2], [None, None, 3]]
    """
    matrix_type = matrix_type or type(matrix)

    try:
        row_type = row_type or type(matrix[0])
    except:
        pass
    if not row_type or row_type is None:
        row_type = list

    try:
        if matrix[0][0] is None:
            value_type = value_type or float
        else:
            value_type = value_type or type(matrix[0][0]) or float
    except:
        pass
    if not value_type or value_type is None:
        value_type = float

    # original matrix is NxM, new matrix will be MxN
    N = len(matrix)
    Ms = [len(row) for row in matrix]
    M = 0 if not Ms else max(Ms)

    ans = []
    # for each row in the new matrix (column in old matrix)
    for j in range(M):
        # add a row full of copies the `fill` value up to the maximum width required
        ans += [row_type([filler] * N)]
        for i in range(N):
            try:
                ans[j][i] = value_type(matrix[i][j])
            except IndexError:
                ans[j][i] = filler
            except TypeError:
                ans[j][i] = filler

    return matrix_type(ans) if isinstance(ans[0], row_type) else matrix_type([row_type(row) for row in ans])


def hist_from_counts(counts, normalize=False, cumulative=False, to_str=False, sep=',', min_bin=None, max_bin=None):
    """Compute an emprical histogram, PMF or CDF in a list of lists

    TESTME: compare results to hist_from_values_list and hist_from_float_values_list
    """
    counters = [dict((i, c)for i, c in enumerate(counts))]

    intkeys_list = [[c for c in counts_dict if (isinstance(c, int) or (isinstance(c, float) and int(c) == c))] for counts_dict in counters]
    min_bin, max_bin = min_bin or 0, max_bin or len(counts) - 1

    histograms = []
    for intkeys, counts in zip(intkeys_list, counters):
        histograms += [OrderedDict()]
        if not intkeys:
            continue
        if normalize:
            N = sum(counts[c] for c in intkeys)
            for c in intkeys:
                counts[c] = float(counts[c]) / N
        if cumulative:
            for i in range(min_bin, max_bin + 1):
                histograms[-1][i] = counts.get(i, 0) + histograms[-1].get(i - 1, 0)
        else:
            for i in range(min_bin, max_bin + 1):
                histograms[-1][i] = counts.get(i, 0)
    if not histograms:
        histograms = [OrderedDict()]

    # fill in the zero counts between the integer bins of the histogram
    aligned_histograms = []

    for i in range(min_bin, max_bin + 1):
        aligned_histograms += [tuple([i] + [hist.get(i, 0) for hist in histograms])]

    if to_str:
        # FIXME: add header row
        return str_from_table(aligned_histograms, sep=sep, max_rows=365 * 2 + 1)

    return aligned_histograms


def hist_from_values_list(values_list, fillers=(None,), normalize=False, cumulative=False, to_str=False, sep=',', min_bin=None, max_bin=None):
    """Compute an emprical histogram, PMF or CDF in a list of lists or a csv string

    Only works for discrete (integer) values (doesn't bin real values).
    `fillers`: list or tuple of values to ignore in computing the histogram

    >>> hist_from_values_list([1,1,2,1,1,1,2,3,2,4,4,5,7,7,9])  # doctest: +NORMALIZE_WHITESPACE
    [(1, 5), (2, 3), (3, 1), (4, 2), (5, 1), (6, 0), (7, 2), (8, 0), (9, 1)]
    >>> hist_from_values_list([(1,9),(1,8),(2,),(1,),(1,4),(2,5),(3,3),(5,0),(2,2)])  # doctest: +NORMALIZE_WHITESPACE
    [[(1, 4), (2, 3), (3, 1), (4, 0), (5, 1)], [(0, 1), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 0), (7, 0), (8, 1), (9, 1)]]
    >>> hist_from_values_list(transposed_matrix([(8,),(1,3,5),(2,),(3,4,5,8)]))  # doctest: +NORMALIZE_WHITESPACE
    [[(8, 1)], [(1, 1), (2, 0), (3, 1), (4, 0), (5, 1)], [(2, 1)], [(3, 1), (4, 1), (5, 1), (6, 0), (7, 0), (8, 1)]]
    """
    value_types = tuple([int, float] + [type(filler) for filler in fillers])

    if all(isinstance(value, value_types) for value in values_list):
        # ignore all fillers and convert all floats to ints when doing counting
        counters = [Counter(int(value) for value in values_list if isinstance(value, (int, float)))]
    elif all(len(row) == 1 for row in values_list) and all(isinstance(row[0], value_types) for row in values_list):
        return hist_from_values_list([values[0] for values in values_list], fillers=fillers, normalize=normalize, cumulative=cumulative,
                                     to_str=to_str, sep=sep, min_bin=min_bin, max_bin=max_bin)
    else:  # assume it's a row-wise table (list of rows)
        return [
            hist_from_values_list(col, fillers=fillers, normalize=normalize, cumulative=cumulative, to_str=to_str, sep=sep,
                                  min_bin=min_bin, max_bin=max_bin)
            for col in transposed_matrix(values_list)
        ]

    if not values_list:
        return []

    intkeys_list = [[c for c in counts if (isinstance(c, int) or (isinstance(c, float) and int(c) == c))] for counts in counters]
    try:
        min_bin = int(min_bin)
    except:
        min_bin = min(min(intkeys) for intkeys in intkeys_list)
    try:
        max_bin = int(max_bin)
    except:
        max_bin = max(max(intkeys) for intkeys in intkeys_list)

    # FIXME: this looks slow and hazardous (like it's ignore min/max bin):
    min_bin = max(min_bin, min((min(intkeys) if intkeys else 0) for intkeys in intkeys_list))  # TODO: reuse min(intkeys)
    max_bin = min(max_bin, max((max(intkeys) if intkeys else 0) for intkeys in intkeys_list))  # TODO: reuse max(intkeys)

    histograms = []
    for intkeys, counts in zip(intkeys_list, counters):
        histograms += [OrderedDict()]
        if not intkeys:
            continue
        if normalize:
            N = sum(counts[c] for c in intkeys)
            for c in intkeys:
                counts[c] = float(counts[c]) / N
        if cumulative:
            for i in range(min_bin, max_bin + 1):
                histograms[-1][i] = counts.get(i, 0) + histograms[-1].get(i - 1, 0)
        else:
            for i in range(min_bin, max_bin + 1):
                histograms[-1][i] = counts.get(i, 0)
    if not histograms:
        histograms = [OrderedDict()]

    # fill in the zero counts between the integer bins of the histogram
    aligned_histograms = []

    for i in range(min_bin, max_bin + 1):
        aligned_histograms += [tuple([i] + [hist.get(i, 0) for hist in histograms])]

    if to_str:
        # FIXME: add header row
        return str_from_table(aligned_histograms, sep=sep, max_rows=365 * 2 + 1)

    return aligned_histograms


def get_similar(obj, labels, default=None, min_similarity=0.5):
    """Similar to fuzzy_get, but allows non-string keys and a list of possible keys

    Searches attributes in addition to keys and indexes to find the closest match.

    See Also:
        `fuzzy_get`

    """
    raise NotImplementedError("Unfinished implementation, needs to be incorporated into fuzzy_get where a list of scores and keywords is sorted.")
    labels = listify(labels)

    def not_found(*args, **kwargs):
        return 0

    min_score = int(min_similarity * 100)
    for similarity_score in [100, 95, 90, 80, 70, 50, 30, 10, 5, 0]:
        if similarity_score <= min_score:
            similarity_score = min_score
        for label in labels:
            try:
                result = obj.get(label, not_found)
            except AttributeError:
                try:
                    result = obj.__getitem__(label)
                except (IndexError, TypeError):
                    result = not_found
            if result is not not_found:
                return result
        if similarity_score == min_score:
            if result is not not_found:
                return result


def normalize_column_labels(obj, labels):
    """Like `get_similar` but returns the matched labels/keys rather than the values and 1 key for each label in labels"""


def update_dict(d, u=None, depth=-1, take_new=True, default_mapping_type=dict, prefer_update_type=False, copy=False):
    """
    Recursively merge (union or update) dict-like objects (Mapping) to the specified depth.

    >>> update_dict({'k1': {'k2': 2}}, {'k1': {'k2': {'k3': 3}}, 'k4': 4})
    {'k1': {'k2': {'k3': 3}}, 'k4': 4}
    >>> update_dict(OrderedDict([('k1', OrderedDict([('k2', 2)]))]), {'k1': {'k2': {'k3': 3}}, 'k4': 4})
    OrderedDict([('k1', OrderedDict([('k2', {'k3': 3})])), ('k4', 4)])
    >>> update_dict(OrderedDict([('k1', dict([('k2', 2)]))]), {'k1': {'k2': {'k3': 3}}, 'k4': 4})
    OrderedDict([('k1', {'k2': {'k3': 3}}), ('k4', 4)])
    >>> orig = {'orig_key': 'orig_value'}
    >>> updated = update_dict(orig, {'new_key': 'new_value'}, copy=True)
    >>> updated == orig
    False
    >>> updated2 = update_dict(orig, {'new_key2': 'new_value2'})
    >>> updated2 == orig
    True
    >>> update_dict({'k1': {'k2': {'k3': 3}}, 'k4': 4}, {'k1': {'k2': 2}}, depth=1, take_new=False)
    {'k1': {'k2': 2}, 'k4': 4}
    >>> update_dict({'k1': {'k2': {'k3': 3}}, 'k4': 4}, None)
    {'k1': {'k2': {'k3': 3}}, 'k4': 4}
    >>> update_dict({'k1': {'k2': {'k3': 3}}, 'k4': 4}, {'k1': ()})
    {'k1': (), 'k4': 4}
    >>> # FIXME: this result is unexpected the same as for `take_new=False`
    >>> update_dict({'k1': {'k2': {'k3': 3}}, 'k4': 4}, {'k1': {'k2': 2}}, depth=1, take_new=True)
    {'k1': {'k2': 2}, 'k4': 4}
    """
    u = u or {}
    orig_mapping_type = type(d)
    if prefer_update_type and isinstance(u, Mapping):
        dictish = type(u)
    elif isinstance(d, Mapping):
        dictish = orig_mapping_type
    else:
        dictish = default_mapping_type
    if copy:
        d = dictish(d)
    for k, v in viewitems(u):
        if isinstance(d, Mapping):
            if isinstance(v, Mapping) and not depth == 0:
                r = update_dict(d.get(k, dictish()), v, depth=max(depth - 1, -1), copy=copy)
                d[k] = r
            elif take_new:
                d[k] = u[k]
        elif take_new:
            d = dictish([(k, u[k])])
    return d


# Fails on py3-style map and list
# def mapped_transposed_lists(lists, default=None):
#     r"""
#     Swap rows and columns in list of lists with different length rows/columns

#     Pattern from
#     http://code.activestate.com/recipes/410687-transposing-a-list-of-lists-with-different-lengths/
#     Replaces any zeros or Nones with default value.

#     Examples:
#     >>> l = mapped_transposed_lists([range(4), [4,5]], None)
#     >>> l
#     [[0, 4], [1, 5], [2, None], [3, None]]
#     >>> mapped_transposed_lists(l)
#     [[0, 1, 2, 3], [4, 5, None, None]]
#     """
#     if not lists:
#         return []
#     # return map(lambda *row: [elem or defval for elem in row], *lists)
#     return list(map(lambda *row: [(el if isinstance(el, (float, int)) else default for el in row], *lists))


def make_name(s, camel=None, lower=None, space='_', remove_prefix=None, language='python', string_type=str):
    """Process a string to produce a valid python variable/class/type name

    Arguments:
      space (str): string to substitute for spaces ('' to delete all whitespace)
      camel (bool): whether to camel-case names, Django Model Name style (first letter capitalized)
      lower (bool): whether to lowercase all strings
      language (str): case-insensitive language identifier (to deterimine allowable identifier characters)
        e.g. 'Python', 'Python2', 'Python3', 'Javascript', 'ECMA'

    Examples:
      Generate Django model names out of file names
      >>> make_name('women in IT.csv', camel=True)
      u'WomenInItCsv'

      Generate Django field names out of CSV header strings
      >>> make_name('ID Number (9-digits)')
      u'id_number_9_digits_'
      >>> make_name("PD / SZ")
      u'pd_sz'

      Generate Javscript object attribute names from CSV header strings
      >>> make_name(u'pi (\u03C0)', space = '', language='javascript')
      u'pi\u03c0'
      >>> make_name(u'pi (\u03C0)', space = '', language='javascript')
      u'pi\u03c0'
    """
    if camel is None and lower is None:
        lower = True
    if not s:
        return None
    ecma_languages = ['ecma', 'javasc']
    unicode_languages = ecma_languages
    language = language or 'python'
    language = language.lower().strip()[:6]
    string_type = string_type or str
    if language in unicode_languages:
        string_type = str
    s = string_type(s)  # TODO: encode in ASCII, UTF-8, or the charset used for this file!
    if remove_prefix and s.startswith(remove_prefix):
        s = s[len(remove_prefix):]
    if camel:
        if space and space == '_':
            space = ''
        if any(c in ' \t\n\r' + string.punctuation for c in s) or s.lower() == s:
            if lower:
                s = s.lower()
            s = s.title()
    elif lower:
        s = s.lower()
    # TODO: add language Regexes to filter characters appropriately for python or javascript
    space_escape = '\\' if space and space not in ' _' else ''
    if language not in ecma_languages:
        invalid_char_regex = re.compile('[^a-zA-Z0-9' + space_escape + space + ']+')
    else:
        # FIXME: Unicode categories and properties only works in Perl Regexes!
        invalid_char_regex = re.compile('[\W' + space_escape + space + ']+', re.UNICODE)
    if space is not None:
        # get rid of all invalid characters, substitting the space-filler for them all
        s = invalid_char_regex.sub(space, s)
        # get rid of duplicate space-filler characters
        if space:
            s = re.sub('[' + space_escape + space + ']{2,}', space, s)
    return s
make_name.DJANGO_FIELD = {'camel': False, 'lower': True, 'space': '_'}
make_name.DJANGO_MODEL = {'camel': True, 'lower': False, 'space': '', 'remove_prefix': 'models'}


def make_filename(s, space=None, language='msdos', strict=False, max_len=None, repeats=1024):
    r"""Process string to remove any characters not allowed by the language specified (default: MSDOS)

    In addition, optionally replace spaces with the indicated "space" character
    (to make the path useful in a copy-paste without quoting).

    Uses the following regular expression to substitute spaces for invalid characters:

        re.sub(r'[ :\\/?*&"<>|~`!]{1}', space, s)

    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', strict=False)
    'Whatever-crazy-s-$h-7-n-m3-ou-can-come-up.-with.-txt-'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', strict=False, repeats=1)
    'Whatever-crazy--s-$h-7-n-m3----ou--can-come-up.-with.-txt--'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', repeats=1)
    'Whatever-crazy--s-$h-7-n-m3----ou--can-come-up.-with.-txt--'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!')
    'Whatever-crazy-s-$h-7-n-m3-ou-can-come-up.-with.-txt-'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', strict=True, repeats=1)
    u'Whatever_crazy_s_h_7_n_m3_ou_can_come_up_with_txt_'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', strict=True, repeats=1, max_len=14)
    u'Whatever_crazy'
    >>> make_filename(r'Whatever crazy &s $h!7 n*m3 ~\/ou/ can come up. with.`txt`!', max_len=14)
    'Whatever-crazy'
    """
    filename = None
    if strict or language.lower().strip() in ('strict', 'variable', 'expression', 'python'):
        if space is None:
            space = '_'
        elif not space:
            space = ''
        filename = make_name(s, space=space, lower=False)
    else:
        if space is None:
            space = '-'
        elif not space:
            space = ''
    if not filename:
        if language.lower().strip() in ('posix', 'unix', 'linux', 'centos', 'ubuntu', 'fedora', 'redhat', 'rhel', 'debian', 'deb'):
            filename = re.sub(r'[^0-9A-Za-z._-]' + '\{1,{0}\}'.format(repeats), space, s)
        else:
            filename = re.sub(r'[ :\\/?*&"<>|~`!]{' + ('1,{0}'.format(repeats)) + r'}', space, s)
    if max_len and int(max_len) > 0 and filename:
        return filename[:int(max_len)]
    else:
        return filename


def update_file_ext(filename, ext='txt', sep='.'):
    r"""Force the file or path str to end with the indicated extension

    Note: a dot (".") is assumed to delimit the extension

    >>> from __future__ import unicode_literals
    >>> update_file_ext('/home/hobs/extremofile', 'bac')
    '/home/hobs/extremofile.bac'
    >>> update_file_ext('/home/hobs/piano.file/', 'music')
    '/home/hobs/piano.file/.music'
    >>> update_file_ext('/home/ninja.hobs/Anglofile', '.uk')
    '/home/ninja.hobs/Anglofile.uk'
    >>> update_file_ext('/home/ninja-corsi/audio', 'file', sep='-')
    '/home/ninja-corsi/audio-file'
    """
    path, filename = os.path.split(filename)

    if ext and ext[0] == sep:
        ext = ext[1:]
    return os.path.join(path, sep.join(filename.split(sep)[:-1 if filename.count(sep) > 1 else 1] + [ext]))


def tryconvert(value, desired_types=SCALAR_TYPES, default=None, empty='', strip=True):
    """
    Convert value to one of the desired_types specified (in order of preference) without raising an exception.

    If value is empty is a string and Falsey, then return the `empty` value specified.
    If value can't be converted to any of the desired_types requested, then return the `default` value specified.

    >>> tryconvert('MILLEN2000', desired_types=float, default='GENX')
    'GENX'
    >>> tryconvert('1.23', desired_types=[int,float], default='default')
    1.23
    >>> tryconvert('-1.0', desired_types=[int,float])  # assumes you want a float if you have a trailing .0 in a str
    -1.0
    >>> tryconvert(-1.0, desired_types=[int,float])  # assumes you want an int if int type listed first
    -1
    >>> repr(tryconvert('1+1', desired_types=[int,float]))
    'None'
    """
    if value in tryconvert.EMPTY:
        if isinstance(value, basestring):
            return type(value)(empty)
        return empty
    if isinstance(value, basestring):
        # there may not be any "empty" strings that won't be caught by the `is ''` check above, but just in case
        if not value:
            return type(value)(empty)
        if strip:
            value = value.strip()
    if isinstance(desired_types, type):
        desired_types = (desired_types,)
    if desired_types is not None and len(desired_types) == 0:
        desired_types = tryconvert.SCALAR
    if len(desired_types):
        if isinstance(desired_types, (list, tuple)) and len(desired_types) and isinstance(desired_types[0], (list, tuple)):
            desired_types = desired_types[0]
        elif isinstance(desired_types, type):
            desired_types = [desired_types]
    for t in desired_types:
        try:
            return t(value)
        except (ValueError, TypeError, InvalidOperation, InvalidContext):
            continue
        # if any other weird exception happens then need to get out of here
        return default
    # if no conversions happened successfully then return the default value requested
    return default
tryconvert.EMPTY = ('', None, float('nan'))
tryconvert.SCALAR = SCALAR_TYPES


def transcode(infile, outfile=None, incoding="shift-jis", outcoding="utf-8"):
    """Change encoding of text file"""
    if not outfile:
        outfile = os.path.basename(infile) + '.utf8'
    with codecs.open(infile, "rb", incoding) as fpin:
        with codecs.open(outfile, "wb", outcoding) as fpout:
            fpout.write(fpin.read())


def strip_br(s):
    r""" Strip the trailing html linebreak character (<BR />) from a string or sequence of strings

    A sequence of strings is assumed to be a row in a CSV/TSV file or words from a line of text
    so only the last element in a sequence is "stripped"

    >>> strip_br(' Title <BR> ')
    ' Title'
    >>> strip_br(list(range(1, 4)))
    [1, 2, 3]
    >>> strip_br((' Column 1<br />', ' Last Column < br / >  '))
    (' Column 1<br />', ' Last Column')
    >>> strip_br(['name', 'rank', 'serial\nnumber', 'date <BR />'])
    ['name', 'rank', 'serial\nnumber', 'date']
    >>> strip_br(None)
    >>> strip_br([])
    []
    >>> strip_br(())
    ()
    >>> strip_br(('one element<br>',))
    ('one element',)
    """

    if isinstance(s, basestring):
        return re.sub(r'\s*<\s*[Bb][Rr]\s*[/]?\s*>\s*$', '', s)
    elif isinstance(s, (tuple, list)):
        # strip just the last element in a list or tuple
        try:
            return type(s)(list(s)[:-1] + [strip_br(s[-1])])
        except:  # len(s) == 0
            return s
    else:
        try:
            return type(s)(strip_br(str(s)))
        except:  # s is None
            return s


def read_csv(csv_file, ext='.csv', format=None, delete_empty_keys=False,
             fieldnames=[], rowlimit=100000000, numbers=False, normalize_names=True, unique_names=True,
             verbosity=0):
    r"""
    Read a csv file from a path or file pointer, returning a dict of lists, or list of lists (according to `format`)

    filename: a directory or list of file paths
    numbers: whether to attempt to convert strings in csv to numbers

    TODO:
        merge with `nlp.util.make_dataframe` function

    Handles unquoted and quoted strings, quoted commas, quoted newlines (EOLs), complex numbers, times, dates, datetimes,
    >>> read_csv(u'"name\r\n",rank,"serial\nnumber",date <BR />\t\n"McCain, John","1","123456789",9/11/2001\n' +
    ...          u'Bob,big cheese,1-23,1/1/2001 12:00 GMT', format='header+values list', numbers=True)
    [[u'name', u'rank', u'serial\nnumber', u'date'], ['McCain, John', 1.0, 123456789.0, '9/11/2001'],
     ['Bob', 'big cheese', '1-23', '1/1/2001 12:00 GMT']]
    """
    if not csv_file:
        return
    if isinstance(csv_file, basestring):
        # truncate `csv_file` in case it is a string buffer containing GBs of data
        path = csv_file[:1025]
        try:
            # see http://stackoverflow.com/a/4169762/623735 before trying 'rU'
            fpin = open(path, 'rUb')  # U = universal EOL reader, b = binary
        except:
            # truncate path more, in case path is used later as a file description:
            path = csv_file[:128]
            fpin = StringIO(str(csv_file))
    else:
        fpin = csv_file
        try:
            path = csv_file.name
        except:
            path = 'unknown file buffer path'

    format = format or 'h'
    format = format[0].lower()

    # if fieldnames not specified then assume that first row of csv contains headings
    csvr = csv.reader(fpin, dialect=csv.excel)
    if not fieldnames:
        while not fieldnames or not any(fieldnames):
            fieldnames = strip_br([str(s).strip() for s in next(csvr)])
        if verbosity > 0:
            logger.info('Column Labels: ' + repr(fieldnames))
    if unique_names:
        norm_names = OrderedDict([(fldnm, fldnm) for fldnm in fieldnames])
    else:
        norm_names = OrderedDict([(num, fldnm) for num, fldnm in enumerate(fieldnames)])
    if normalize_names:
        norm_names = OrderedDict([(num, make_name(fldnm, **make_name.DJANGO_FIELD)) for num, fldnm in enumerate(fieldnames)])
        # required for django-formatted json files
        model_name = make_name(path, **make_name.DJANGO_MODEL)
    if format in 'c':  # columnwise dict of lists
        recs = OrderedDict((norm_name, []) for norm_name in list(norm_names.values()))
    elif format in 'vh':
        recs = [fieldnames]
    else:
        recs = []
    if verbosity > 0:
        logger.info('Field Names: ' + repr(norm_names if normalize_names else fieldnames))
    rownum = 0
    eof = False
    pbar = None
    start_seek_pos = fpin.tell() or 0
    if verbosity > 1:
        print('Starting at byte {} in file buffer.'.format(start_seek_pos))
    fpin.seek(0, os.SEEK_END)
    file_len = fpin.tell() - start_seek_pos  # os.fstat(fpin.fileno()).st_size
    fpin.seek(start_seek_pos)

    if verbosity > 1:
        print('There appear to be {} bytes remaining in the file buffer. Resetting (seek) to starting position in file.'.format(file_len))
    if verbosity > 0:
        pbar = progressbar.ProgressBar(maxval=file_len)
        pbar.start()
    while csvr and rownum < rowlimit and not eof:
        if pbar:
            pbar.update(fpin.tell() - start_seek_pos)
        rownum += 1
        row = []
        row_dict = OrderedDict()
        # skip rows with all empty strings as values,
        while not row or not any(len(x) for x in row):
            try:
                row = next(csvr)
                if verbosity > 1:
                    logger.info('  row content: ' + repr(row))
            except StopIteration:
                eof = True
                break
        if eof:
            break
        if len(row) and isinstance(row[-1], basestring) and len(row[-1]):
            row = strip_br(row)
        if numbers:
            # try to convert the type to a numerical scalar type (int, float etc)
            row = [tryconvert(v, desired_types=NUMBERS_AND_DATETIMES, empty=None, default=v) for v in row]
        if row:
            N = min(max(len(row), 0), len(norm_names))
            row_dict = OrderedDict(
                ((field_name, field_value) for field_name, field_value in zip(
                    list(list(norm_names.values()) if unique_names else norm_names)[:N], row[:N])
                    if (str(field_name).strip() or delete_empty_keys is False))
            )
            if format in 'dj':  # django json format
                recs += [{"pk": rownum, "model": model_name, "fields": row_dict}]
            elif format in 'vhl':  # list of lists of values, with header row (list of str)
                recs += [[value for field_name, value in viewitems(row_dict) if (field_name.strip() or delete_empty_keys is False)]]
            elif format in 'c':  # columnwise dict of lists
                for field_name in row_dict:
                    recs[field_name] += [row_dict[field_name]]
                if verbosity > 2:
                    print([recs[field_name][-1] for field_name in row_dict])
            else:
                recs += [row_dict]
            if verbosity > 2 and format not in 'c':
                print(recs[-1])

    if file_len > fpin.tell():
        logger.info("Only %d of %d bytes were read and processed." % (fpin.tell(), file_len))
    if pbar:
        pbar.finish()
    fpin.close()
    if not unique_names:
        return recs, norm_names
    return recs


# date and datetime separators
COLUMN_SEP = re.compile(r'[,/;]')


class Object(object):
    """If your dict is "flat", this is a simple way to create an object from a dict

    >>> obj = Object()
    >>> obj.__dict__ = {'a': 1, 'b': 2}
    >>> obj.a, obj.b
    (1, 2)
    """
    pass


# For a nested dict, you need to recursively update __dict__
def dict2obj(d):
    """Convert a dict to an object or namespace


    >>> d = {'a': 1, 'b': {'c': 2}, 'd': ["hi", {'foo': "bar"}]}
    >>> obj = dict2obj(d)
    >>> obj.b.c
    2
    >>> obj.d
    ['hi', {'foo': 'bar'}]
    >>> d = {'a': 1, 'b': {'c': 2}, 'd': [("hi", {'foo': "bar"})]}
    >>> obj = dict2obj(d)
    >>> obj.d.hi.foo
    'bar'
    """
    if isinstance(d, (Mapping, list, tuple)):
        try:
            d = dict(d)
        except (ValueError, TypeError):
            return d
    else:
        return d
    obj = Object()
    for k, v in viewitems(d):
        obj.__dict__[k] = dict2obj(v)
    return obj


def any_generated(gen):
    """like `any` but returns False for empty generators
    >>> any_generated((v for v in (0,object())))
    True
    >>> any_generated((v for v in (0,False)))
    False
    >>> any_generated((v for v in ()))
    False
    """
    for v in gen:
        if bool(v):
            return True
    return False


def make_series(x, *args, **kwargs):
    """Coerce a provided array/sequence/generator into a pandas.Series object
    FIXME: Deal with CSR, COO, DOK and other sparse matrices like this:
       pd.Series(csr.toarray()[:,0])
         or, if csr.shape[1] == 2
       pd.Series(csr.toarray()[:,1], index=csr.toarray()[:,0])
    >>> make_series(range(1, 4))
    0    1
    1    2
    2    3
    dtype: int64
    >>> make_series(xrange(1, 4))
    0    1
    1    2
    2    3
    dtype: int64
    >>> make_series(list('ABC'))
    0    A
    1    B
    2    C
    dtype: object
    >>> make_series({'a': .8, 'be': .6}, name=None)
    a     0.8
    be    0.6
    dtype: float64
    """
    if isinstance(x, pd.Series):
        return x
    try:
        if len(args) == 1 and 'pk' not in args:
            # args is a tuple, so needs to be turned into a list to prepend pk for Series index
            args = ['pk'] + list(args)
        df = pd.DataFrame.from_records(getattr(x, 'objects', x).values(*args))
        if len(df.columns) == 1:
            return df[df.columns[0]]
        elif len(df.columns) >= 2:
            return df.set_index(df.columns[0], drop=False)[df.columns[1]]
        logger.warn('Unable to coerce {} into a pd.Series using args {} and kwargs {}.'.format(x, args, kwargs))
        return pd.Series()
    except (AttributeError, TypeError):
        kwargs['name'] = getattr(x, 'name', None) if 'name' not in kwargs else kwargs['name']
        if 'index' in kwargs:
            x = list(x)
        try:
            return pd.Series(x, **kwargs)
        except:
            logger.debug(format_exc())
            try:
                return pd.Series(np.array(x), **kwargs)
            except:
                logger.debug(format_exc())
                return pd.Series(x, **kwargs)


def encode(obj):
    r"""Encode all unicode/str objects in a dataframe in the encoding indicated (as a fun attribute)
    similar to to_ascii, but doesn't return a None, even when it fails.
    >>> encode(u'Is 2013 a year or a code point in the NeoMatch strings "\u2013"?')
    'Is 2013 a year or a code point in the NeoMatch strings "\xe2\x80\x93"?'
    """
    try:
        return obj.encode(encode.encoding)
    except AttributeError:
        pass
    except UnicodeDecodeError:
        logger.warning('Problem with byte sequence of type {}.'.format(type(obj)))
        # TODO: Check PG for the proper encoding and fix Django ORM settings so that unicode can be UTF-8 encoded!
        return ''.join([c for c in obj if c < MAX_CHR])
    # TODO: encode sequences of strings and dataframes of strings
    return obj
encode.encoding = 'utf-8'


def clean_series(series, *args, **kwargs):
    """Ensure all datetimes are valid Timestamp objects and dtype is np.datetime64[ns]
    >>> from datetime import timedelta
    >>> clean_series(pd.Series([datetime.datetime(1, 1, 1), 9, '1942', datetime.datetime(1970, 10, 23)]))
    0    1677-09-22 00:12:44+00:00
    1                            9
    2                         1942
    3    1970-10-23 00:00:00+00:00
    dtype: object
    >>> clean_series(pd.Series([datetime.datetime(1, 1, 1), datetime.datetime(3000, 10, 23)]))
    0             1677-09-22 00:12:44+00:00
    1   2262-04-11 23:47:16.854775807+00:00
    dtype: datetime64[ns, UTC]
    """
    if not series.dtype == np.dtype('O'):
        return series
    if any_generated((isinstance(v, datetime.datetime) for v in series)):
        series = series.apply(clip_datetime)
    if any_generated((isinstance(v, basestring) for v in series)):
        series = series.apply(encode)
    return series


def make_dataframe(table, clean=True, verbose=False, **kwargs):
    """Coerce a provided table (QuerySet, list of lists, list of Series)
    >>> dt = datetime.datetime
    >>> make_dataframe([[1,2,3],[4,5,6]])
       0  1  2
    0  1  2  3
    1  4  5  6
    >>> make_dataframe([])
    Empty DataFrame
    Columns: []
    Index: []
    >>> make_dataframe([{'a': 2, 'b': 3}, PrettyDict([('a', 4), ('b', 5)])])
       a  b
    0  2  3
    1  4  5
    >>> make_dataframe([[dt(2700, 1, 1), dt(2015, 11, 2)], [(2700 - 2015) * 365.25 + 60, 1]]).T
                                         0       1
    0  2262-04-11 23:47:16.854775807+00:00  250256
    1            2015-11-02 00:00:00+00:00       1
    """
    if hasattr(table, 'objects') and not callable(table.objects):
        table = table.objects
    if hasattr(table, 'filter') and callable(table.values):
        table = pd.DataFrame.from_records(list(table.values()).all())
    elif isinstance(table, basestring) and os.path.isfile(table):
        table = pd.DataFrame.from_csv(table)
    # elif isinstance(table, ValuesQuerySet) or (isinstance(table, (list, tuple)) and
    #                                            len(table) and all(isinstance(v, Mapping) for v in table)):
    #     table = pd.DataFrame.from_records(table)
    try:
        table = pd.DataFrame(table, **kwargs)
    except:
        table = pd.DataFrame(table)
    if clean and len(table) and isinstance(table, pd.DataFrame):
        if verbose:
            print('Cleaning up OutOfBoundsDatetime values...')
        for col in table.columns:
            if any_generated((isinstance(v, DATETIME_TYPES) for v in table[col])):
                table[col] = clean_series(table[col])
        table = table.dropna(how='all')
    return table
    # # in case the args and kwargs are intended for pd.DataFrame constructor rather than make_dataframe
    # return pd.DataFrame(table, **kwargs)


def column_name_to_date(name):
    """
    TODO: should probably assume a 2000 epoch for 2-digit dates

    >>> column_name_to_date('10-Apr')
    datetime.date(10, 4, 1)
    >>> column_name_to_date('10_2011')
    datetime.date(2011, 10, 1)
    >>> column_name_to_date('apr_10')
    datetime.date(10, 4, 1)
    """
    month_nums = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    year_month = re.split(r'[^0-9a-zA-Z]{1}', name)
    try:
        year = int(year_month[0])
        month = year_month[1]
    except:
        year = int(year_month[1])
        month = year_month[0]
    month = month_nums.get(str(month).lower().title(), None)
    if 0 <= year <= 2100 and 1 <= month <= 12:
        return datetime.date(year, month, 1)
    try:
        year = int(year_month[1])
        month = int(year_month[0])
    except:
        year. month = 0, 0
    if 0 <= year <= 2100 and 1 <= month <= 12:
        return datetime.date(year, month, 1)
    try:
        month = int(year_month[1])
        year = int(year_month[0])
    except:
        year. month = 0, 0
    if 0 <= year <= 2100 and 1 <= month <= 12:
        return datetime.date(year, month, 1)


def first_digits(s, default=0):
    """Return the fist (left-hand) digits in a string as a single integer, ignoring sign (+/-).
    >>> first_digits('+123.456')
    123
    """
    s = re.split(r'[^0-9]+', str(s).strip().lstrip('+-' + charlist.whitespace))
    if len(s) and len(s[0]):
        return int(s[0])
    return default


def int_pair(s, default=(0, None)):
    """Return the digits to either side of a single non-digit character as a 2-tuple of integers

    >>> int_pair('90210-007')
    (90210, 7)
    >>> int_pair('04321.0123')
    (4321, 123)
    """
    s = re.split(r'[^0-9]+', str(s).strip())
    if len(s) and len(s[0]):
        if len(s) > 1 and len(s[1]):
            return (int(s[0]), int(s[1]))
        return (int(s[0]), default[1])
    return default


def make_us_postal_code(s, allowed_lengths=(), allowed_digits=()):
    """
    >>> make_us_postal_code(1234)
    '01234'
    >>> make_us_postal_code(507.6009)
    '507'
    >>> make_us_postal_code(90210.0)
    '90210'
    >>> make_us_postal_code(39567.7226)
    '39567-7226'
    >>> make_us_postal_code(39567.7226)
    '39567-7226'
    """
    allowed_lengths = allowed_lengths or tuple(N if N < 6 else N + 1 for N in allowed_digits)
    allowed_lengths = allowed_lengths or (2, 3, 5, 10)
    ints = int_pair(s)
    z = str(ints[0]) if ints[0] else ''
    z4 = '-' +