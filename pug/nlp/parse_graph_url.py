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
    node_names = list(set(node_names_only))
    new_nodes, old_nodes = {}, {}
    for new, old in enumerate(node_names):
        new_nodes[old] = new
        old_nodes[new] = old
    return old_nodes, new_nodes


def node_name_lists(edge_list):
    """
    Return 2 lists that retain the order of nodes mentioned in the edges list: a list of full names and a list of cleaned names.

    node_name_lists([{'source': 'Origin,2.7, 3 ', 'target': 'Destination,1,2', 'value': 9}, {'source': 'Origin,4', 'target': 'New', 'value': 1}])
    (['Origin,2.7, 3 ', 'Destination,1,2', 'New'], ['Origin', 'Destination', 'New'])
    """
    node_names_only, node_full_names = [], []
    for edge in edge_list:
        node_full_names += [edge['source'], edge['target']]
        node_names_only += [node_name(node_full_names[-2])['name'], node_name(node_full_names[-1])['name']]
    for_del = []
    for i, name in enumerate(node_names_only):
        if name in node_names_only[:i]:
            for_del += [i]
    for i in reversed(for_del):
        del(node_full_names[i])
        del(node_names_only[i])
    return node_full_names, node_names_only


def node_names(edges):
    """Parse the node names found in a graph definition string

    >>> node_names([{'source': 'Origin,2.7, 3 ', 'target': 'Destination,1,2', 'value': 9}, {'source': 'Origin,4', 'target': 'New', 'value': 1}])
    [{'charge': 4.0, 