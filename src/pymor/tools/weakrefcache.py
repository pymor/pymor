# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import weakref


class WeakRefCache:
    """Simple WeakKeyDictionary-like cache.

    This class allows storing additional data associated with any weakref-able
    object. The data is removed when the corresponding object dies.
    Compared to WeakKeyDictionary, object do not need to implement hashablel
    or comparable.

    See https://github.com/python/cpython/issues/88306 for further context.
    The implementation here can be rather simple since we do not support iteration.
    """

    def __init__(self):
        self.data = {}

    def set(self, key, value):
        i = id(key)
        ref = weakref.ref(key, lambda w: self._remove(i))
        self.data[i] = (ref, value)

    def get(self, key):
        ref, value = self.data[id(key)]
        if ref() is None:
            raise KeyError  # under normal circumstances, this should never happen.
                            # the entry should be removed as soon as the object dies
        return value

    def _remove(self, i):
        del self.data[i]
