
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Crawlers and Scrapers for retrieving data/tables from URLs."""
from __future__ import division, print_function, absolute_import
from past.builtins import basestring

import os
import datetime
import logging
from collections import OrderedDict

import pandas as pd
# from bs4 import BeautifulSoup
# import urllib2

from pug.nlp.regex import email_popular, cre_url
from pug.nlp.util import make_name
from pug.nlp.constant import DATA_PATH
from pug.nlp.db import strip_nonascii

log = logging.getLogger(__name__)


def find_emails(html=os.path.join(DATA_PATH, 'Locations.html')):
    """Extract email addresses from an html page or ASCII stream."""
    if isinstance(html, (str, bytes)):
        if os.path.isfile(html):
            html = open(html, 'r').read()
        html = email_popular.findall(html)
    return [x[0] for x in html]


uni_ascii = OrderedDict([
    (u'\xc2\xa0', ' '),      # nonbreaking? space:     " "
    (u'\xe2\x80\x91', '-'),  # smaller dash shifted left:     "‑"
    (u'\xe3\x81\xa3', '>'),  # backward skewed subscripted C: "っ"
    (u'\xc2\xa0', 'A'),     # Angstrom symbol
    (u'\u2011', '-'),      # smaller dash shifted left:     "‑"
    (u'\u3063', '>'),      # backward skewed subscripted C: "っ"
    (u'\xa0', ' '),         # nonbreaking? space:     " "
    ])


spaced_uni_emoticons = OrderedDict([
    # lenny
    (u'( ͡° ͜ʖ ͡°)', '(^-_-^)'),
    (u'( ͡°͜ ͡°)', '(^-_-^)'),
    (u'(͡° ͜ʖ ͡°)', '(^-_-^)'),
    (u'(͡°͜ ͡°)', '(^-_-^)'),
    # kiss
    (u"( '}{' )", "(_'}{'_)"),
    # lenny
    (u'( \xcd\xa1\xc2\xb0 \xcd\x9c\xca\x96 \xcd\xa1\xc2\xb0)', '(^-_-^)'),
    (u'( \xcd\xa1\xc2\xb0\xcd\x9c \xcd\xa1\xc2\xb0)', '(^-_-^)'),
    ])

spaced_ascii_emoticons = OrderedDict([
    (u"( '}{' )", "(_'}{'_)"),
    (u'( \xcd\xa1\xc2\xb0 \xcd\x9c\xca\x96 \xcd\xa1\xc2\xb0)', '(^-_-^)'),  # Lenny
    (u'( \xcd\xa1\xc2\xb0\xcd\x9c \xcd\xa1\xc2\xb0)', '(^-_-^)'),  # Lenny
    ])


def transcode_unicode(s):
    print(s)
    try:
        s = unicode(s).encode('utf-8')
    except:
        pass
    try:
        s = str(s).decode('utf-8')
    except:
        pass
    for c, equivalent in uni_ascii.iteritems():
        print(c)
        print(type(c))
        uni = unicode(s).replace(c, equivalent)
    return strip_nonascii(uni)


def clean_emoticon_wiki_table(html='https://en.wikipedia.org/wiki/List_of_emoticons',
                              save='list_of_emoticons-wikipedia-cleaned.csv',
                              data_dir=DATA_PATH,
                              table_num=1,
                              **kwargs):
    wikitables = pd.read_html(html, header=0)
    for i, wikidf in enumerate(wikitables):
        header = (' '.join(str(s).strip() for s in wikidf.columns)).lower()
        if table_num == i or (table_num is None and 'meaning' in header):
            break
    df = wikidf
    df.columns = [make_name(s, lower=True) for s in df.columns]
    table = []
    for icon, meaning in zip(df[df.columns[0]], df[df.columns[1]]):
        # kissing couple has space in it
        for ic, uni_ic in spaced_uni_emoticons.iteritems():
            icon = icon.replace(ic, uni_ic)
        for ic, asc_ic in spaced_ascii_emoticons.iteritems():