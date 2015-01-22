# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

# {{{ http://code.activestate.com/recipes/577504/ (r3) MIT licensed

from __future__ import absolute_import, division, print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import resource
try:
    from reprlib import repr
except ImportError:
    pass


last_memory_usage = 0


def memory_usage(unit='mb'):
    """Returns the memory usage of the current process in bytes

    Returns
    -------
    usage
        Current memory usage.
    change
        Change of memory usage since last call.
    """
    unit = unit.lower()
    assert unit in ('b', 'kb', 'mb', 'gb')
    factors = {'b': 1,
               'kb': 1024,
               'mb': 1024**2,
               'gb': 1024**3}
    global last_memory_usage
    x = last_memory_usage
    last_memory_usage = resource.getrusage(resource.RUSAGE_SELF)[2] * 1024
    return (last_memory_usage / factors[unit], (last_memory_usage - x) / factors[unit])


def print_memory_usage(msg=None, unit='mb'):
    u = memory_usage(unit)
    if msg is None:
        print('Memory usage {0:5.1f} {1} - delta: {2:5.1f} {1}'.format(u[0], unit.upper(), u[1]))
    else:
        print('Memory usage {0:5.1f} {1} - delta: {2:5.1f} {1} - {3}'.format(u[0], unit.upper(), u[1], msg))


def total_size(o, handlers=None, verbose=False):
    """ Returns the approximate memory footprint of an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    handlers = handlers if handlers else {}
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


# #### Example call #####

if __name__ == '__main__':
    d = dict(a=1, b=2, c=3, d=[4, 5, 6, 7], e='a string of chars')
    print(total_size(d, verbose=True))
# # end of http://code.activestate.com/recipes/577504/ }}}
