"""Parse a URL that describes a graph (connection or network diagram)

For use with a REST interface to display D3.js force-directed graphs
"""

from collections import Mapping
from pug.nlp.util import intify, listify


def node_name(name, use_defaults=False):
    """
    >>> sorted(node_name('Origin,2.7, 3 ')[1].items())
    [('charge', 2.7), ('group', 3), ('name', 'Origin')]
    >>> node_name('Origin,2.7, 3 ')[0]
    'Origin'
    """
    # if the name is not a string, but a dict defining a node, then just set the defaults and return it
    if isinstance(name, Mapping):
        ans = dict(name)
        for j, fi