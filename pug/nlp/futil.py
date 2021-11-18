#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""file utils"""
from __future__ import division, print_function, absolute_import
from builtins import str  # noqa
from past.builtins import basestring  # noqa
try:  # python 3.5+
    from io import StringIO
    from ConfigParser import ConfigParser
    from itertools import izip as zip
except:
    from StringIO import StringIO
    from configparser import ConfigParser

import os
import datetime
import warnings
import subprocess
from collections import Mappin