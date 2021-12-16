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
        for j, field in enumerate(node_name.schema):
            if field['key'] not in ans:
                ans[field['key']] = field['default']
        return ans
    seq = listify(name, delim=',')
    ans = {}
    for j, field in enumerate(node_name.schema):
        if 'default' in field:
            try:
                ans[field['key']] = field['type'](seq[j])
            except:
                if use_defaults:
                    ans[field['key']] = field['default']
        else:
            try:
                ans[field['key']] = ans.get(field['key'], field['type'](seq[j]))
            except:
                pass
    return ans
node_name.schema = (
                {'key': 'name', 'type': str},  # TODO: use the absence of a default value (rather than index > 0) to identify mandatory fields
                {'key': 'charge', 'type': float, 'default': 1},
                {'key': 'group', 'type': intify, 'default': 0},  # TODO: this should be a string like the names/indexes to nodes (groups are just hidden nodes)
              )


def node_name_dictionaries(edges):
    """
    Return 2 dictionaries that translate from the cleaned/striped name to fully qualified node names, and vice versa.
    """
    node_names_only = []
    for edge in edges:
        node_names_only += [node_name(edge['source'])['name'], node_name(edge['target'])['name']]
    node_names = list(