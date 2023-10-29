
# python-nlp-toolkit

[![Build Status](https://travis-ci.org/totalgood/python-nlp-toolkit.svg?branch=master "Travis Build & Test Status")](https://travis-ci.org/totalgood/python-nlp-toolkit)
[![Coverage Status](https://coveralls.io/repos/totalgood/python-nlp-toolkit/badge.png)](https://coveralls.io/r/totalgood/python-nlp-toolkit)
[![Latest Release Version](https://badge.fury.io/py/python-nlp-toolkit.svg)](https://pypi.python.org/pypi/python-nlp-toolkit/)
<!-- [![Downloads](https://pypip.in/d/python-nlp-toolkit/badge.png)](https://pypi.python.org/pypi/python-nlp-toolkit/) -->

## python-nlp-toolkit Utilities

This section of the python namespace package consists of natural language processing (NLP) and text processing utilities built for Python User Communities.

---

## Installation

### On a Posix System

Would you like to contribute?

    git clone https://github.com/totalgood/python-nlp-toolkit.git

If you're a user, not a developer, and you have an up-to-date posix OS with the postgres, xml2, and xlst development packages installed, then simply use `pip`.

    pip install python-nlp-toolkit

### Fedora

If you're on Fedora >= 16 but haven't done much python binding development, then install some libraries before pip will succeed.

    sudo yum install -y python-devel libxml2-devel libxslt-devel gcc-gfortran python-scikit-learn postgresql postgresql-server postgresql-libs postgresql-devel
    pip install python-nlp-toolkit

### Bleeding Edge

While the releases can be unstable, if you want to experiment with the latest, untested code:

    pip install git+git://github.com/jawking/python-nlp-toolkit.git@master

### Warning

This software is in alpha testing.  Install at your own risk.

---

## Development

PR contributions are always welcome and contributors will be added to the `__authors__` list:

    git clone https://github.com/totalgood/python-nlp-toolkit.git
