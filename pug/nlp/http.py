"""Functions for manipulating and composing http-protocol traffic

* simplify_get: eliminate empty and/or redundant HTTP GET parameters from a request dict

"""
import datetime


def simplify_get(get_dict, keys_to_del=None, datetime_to_date=True):
    """Delete any GET request key/