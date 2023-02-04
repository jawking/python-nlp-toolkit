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

    >>> hist_fro