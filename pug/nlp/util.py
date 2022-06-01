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
from 