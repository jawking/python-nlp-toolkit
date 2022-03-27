#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""table utils"""
from __future__ import division, print_function, absolute_import
from builtins import str  # , unicode  # noqa
# from future.utils import viewitems  # noqa
from past.builtins import basestring
# try:  # python 3.5+
#     from io import StringIO
#     from ConfigParser import ConfigParser
#     from itertools import izip as zip
# except:
#     from StringIO import StringIO
#     from configparser import ConfigParser

from types import NoneType

from traceback import print_exc
import datetime
import xlrd
import pandas as pd
from dateutil.parser import parse as parse_date
from .futil import find_files


def dataframe_from_excel(path, sheetname=0, header=0, skiprows=None):  # , parse_dates=False):
    """Thin wrapper for pandas.io.excel.read_excel() that accepts a file path and sheet index/name

    Arguments:
      path (str): file or folder to retrieve CSV files and `pandas.DataFrame`s from
      ext (str): file name extension (to filter files by)
      date_parser (function): if the MultiIndex can 