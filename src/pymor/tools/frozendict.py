# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

# The following implementation is based on
# https://code.activestate.com/recipes/414283-frozen-dictionaries/


class FrozenDict(dict):
    """An immutable dictionary."""

    __slots__ = ()

    @property
    def _blocked_attribute(self):
        raise AttributeError(f'A {type(self).__name__} cannot be modified.')

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute

    def __new__(cls, *args, **kwargs):
        new = dict.__new__(cls)
        dict.__init__(new, *args, **kwargs)
        new._post_init()
        return new

    def __init__(self, *args, **kwargs):
        # ensure that dict cannot be modified by calling __init__
        pass

    def _post_init(self):
        pass

    def __repr__(self):
        return f'FrozenDict({dict.__repr__(self)})'

    def __reduce__(self):
        return (type(self), (dict(self),))


class SortedFrozenDict(FrozenDict):
    """A sorted immutable dictionary."""

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        new = dict.__new__(cls)
        # the following only works on Python >= 3.7 or CPython >= 3.6
        dict.__init__(new, sorted(dict(*args, **kwargs).items()))
        new._post_init()
        return new

    def __repr__(self):
        return f'SortedFrozenDict({dict.__repr__(self)})'
