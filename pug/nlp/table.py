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
      date_parser (function): if the MultiIndex can be interpretted as a datetime, this parser will be used

    Returns:
      dict of DataFrame: { file_path: flattened_data_frame }
    """
    sheetname = sheetname or 0
    if isinstance(sheetname, (basestring, float)):
        try:
            sheetname = int(sheetname)
        except (TypeError, ValueError, OverflowError):
            sheetname = str(sheetname)
    wb = xlrd.open_workbook(path)
    # if isinstance(sheetname, int):
    #     sheet = wb.sheet_by_index(sheetname)
    # else:
    #     sheet = wb.sheet_by_name(sheetname)
    # assert(not parse_dates, "`parse_dates` argument and function not yet implemented!")
    # table = [sheet.row_values(i) for i in range(sheet.nrows)]
    return pd.io.excel.read_excel(