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
from collections import Mapping
import errno


def walk_level(path, level=1):
    """Like os.walk, but takes `level` kwarg that indicates how deep the recursion will go.

    Notes:
      TODO: refactor `level`->`depth`

    References:
      http://stackoverflow.com/a/234329/623735

    Args:
     path (str):  Root path to begin file tree traversal (walk)
      level (int, optional): Depth of file tree to halt recursion at.
        None = full recursion to as deep as it goes
        0 = nonrecursive, just provide a list of files at the root level of the tree
        1 = one level of depth deeper in the tree

    Examples:
      >>> root = os.path.dirname(__file__)
      >>> all((os.path.join(base,d).count('/')==(root.count('/')+1)) for (base, dirs, files) in walk_level(root, level=0) for d in dirs)
      True
    """
    if level is None:
        level = float('inf')
    path = path.rstrip(os.path.sep)
    if os.path.isdir(path):
        root_level = path.count(os.path.sep)
        for root, dirs, files in os.walk(path):
            yield root, dirs, files
            if root.count(os.path.sep) >= root_level + level:
                del dirs[:]
    elif os.path.isfile(path):
        yield os.path.dirname(path), [], [os.path.basename(path)]
    else:
        raise RuntimeError("Can't find a valid folder or file for path {0}".format(repr(