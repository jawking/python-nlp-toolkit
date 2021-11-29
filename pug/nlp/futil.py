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
        raise RuntimeError("Can't find a valid folder or file for path {0}".format(repr(path)))


def path_status(path, filename='', status=None, verbosity=0):
    """ Retrieve the access, modify, and create timetags for a path along with its size

    Arguments:
        path (str): full path to the file or directory to be statused
        status (dict): optional existing status to be updated/overwritten with new status values

    Returns:
        dict: {'size': bytes (int), 'accessed': (datetime), 'modified': (datetime), 'created': (datetime)}
    """
    status = status or {}
    if not filename:
        dir_path, filename = os.path.split()  # this will split off a dir and as `filename` if path doesn't end in a /
    else:
        dir_path = path
    full_path = os.path.join(dir_path, filename)
    if verbosity > 1:
        print(full_path)
    status['name'] = filename
    status['path'] = full_path
    status['dir'] = dir_path
    status['type'] = []
    try:
        status['size'] = os.path.getsize(full_path)
        status['accessed'] = datetime.datetime.fromtimestamp(os.path.getatime(full_path))
        status['modified'] = datetime.datetime.fromtimestamp(os.path.getmtime(full_path))
        status['created'] = datetime.datetime.fromtimestamp(os.path.getctime(full_path))
        status['mode'] = os.stat(full_path).st_mode   # first 3 digits are User, Group, Other permissions: 1=execute,2=write,4=read
        if os.path.ismount(full_path):
            status['type'] += ['mount-point']
        elif os.path.islink(full_path):
            status['type'] += ['symlink']
        if os.path.isfile(full_path):
            status['type'] += ['file']
        elif os.path.isdir(full_path):
            status['type'] += ['dir']
        if not status['type']:
            if os.stat.S_ISSOCK(status['mode']):
                status['type'] += ['socket']
            elif os.stat.S_ISCHR(status['mode']):
                status['type'] += ['special']
            elif os.stat.S_ISBLK(status['mode']):
                status['type'] += ['block-device']
            elif os.stat.S_ISFIFO(status['mode']):
                status['type'] += ['pipe']
        if not status['type']:
            status['type'] += ['unknown']
        elif status['type'] and status['type'][-1] == 'symlink':
            status['type'] += ['broken']
    except OSError:
        status['type'] = ['nonexistent'] + status['type']
        if verbosity > -1:
            warnings.warn("Unable to stat path '{}'".format(full_path))
    status['type'] = '->'.join(status['type'])

    return status


def find_files(path='', ext='', level=None, typ=list, dirs=False, files=True, verbosity=0):
    """ Recursively find all files in the indicated directory

    Filter by the indicated file name extension (ext)

    Args:
      path (str):  Root/base path to search.
      ext (str):   File name extension. Only file paths that ".endswith()" this string will be returned
      level (int, optional): Depth of file tree to halt recursion at.
        None = full recursion to as deep as it goes
        0 = nonrecursive, just provide a list of files at the root level of the tree
        1 = one level of depth deeper in the tree
      typ (type):  output type (default: list). if a mapping type is provided the keys will be the full paths (unique)
      dirs (bool):  Whether to yield dir paths along with file paths (default: False)
      files (bool): Whether to yield file paths (default: True)
        `dirs=True`, `files=False` is equivalent to `ls -d`

    Returns:
      list of dicts: dict keys are { 'path', 'name', 'bytes', 'created', 'modified', 'accessed', 'permissions' }
        path (str): Full, absolute paths to file beneath the indicated directory and ending with `ext`
        name (str): File name only (everythin after the last slash in the path)
        size (int): File size in bytes
        created (datetime): File creation timestamp from file system
        modified (datetime): File modification timestamp from file system
        accessed (datetime): File access timestamp from file system
        permissions (int): File permissions bytes as a chown-style integer with a maximum of 4 digits
        type (str): One of 'file', 'dir', 'symlink->file', 'symlink->dir', 'symlink->broken'
          e.g.: 777 or 1755

    Examples:
      >>> 'util.py' in [d['name'] for d in find_files(os.path.dirname(__file__), ext='.py', level=0)]
      True
      >>> (d for d in find_files(os.path.dirname(__file__), ext='.py') if d['name'] == 'util.py').next()['size'] > 1000
      True

      There should be an __init__ file in the same directory as this script.
      And it should be at the top of the list.
      >>> sorted(d['name'] for d in find_files(os.path.dirname(__file__), ext='.py', level=0))[0]
      '__init__.py'
      >>> all(d['type'] in ('file','dir','symlink->file','symlink->dir','mount-point->file','mount-point->dir','block-device',
                            'symlink->broken','pipe','special','socket','unknown') for d in find_files(level=1, files=True, dirs=True))
      True
      >>> os.path.join(os.path.dirname(__file__), '__init__.py') in find_files(
      ... os.path.dirname(__file__), ext='.py', level=0, typ=dict)
      True
    """
    gen = generate_files(path, ext=ext, level=level, dirs=dirs, files=files, verbosity=verbosity)
    if isinstance(typ(), Mapping):
        return typ((ff['path'], ff) for ff in gen)
    elif typ is not None:
        return typ(gen)
    else:
        return gen


def generate_files(path='