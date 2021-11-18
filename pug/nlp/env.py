"""Keep this as simple as possible to minimize the possability of error when used within a django settings.py file"""

import sys
import os

def get(var_name, default=False, verbosity=0):
    """ Get the environment variable or assume a default, but let the user know about the error."""
    try:
        value = os.environ[var_name]
        if str(value).strip().lower() in ['false', 'no', '