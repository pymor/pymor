# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import contextlib
import tempfile
import time
import os
from uuid import uuid4
from datetime import datetime, timedelta
import numpy as np
import pytest

from pymor.core import cache
from pymor.models.basic import StationaryModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymortests.base import runmodule


SLEEP_DELTA = timedelta(milliseconds=200)


class IamMemoryCached(cache.CacheableObject):

    def __init__(self):
        self.cache_region = 'memory'

    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(SLEEP_DELTA.total_seconds())
        return arg


class IamDiskCached(cache.CacheableObject):

    def __init__(self):
        self.cache_region = 'disk'

    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(SLEEP_DELTA.total_seconds())
        return arg


class IamLimitedCached(cache.CacheableObject):

    def __init__(self, cache_region='disk'):
        self.cache_region = cache_region

    @cache.cached
    def me_takey_no_time(self, arg):
        return int(arg)


class IWillBeCopied(cache.CacheableObject):

    def __init__(self):
        super().__init__()

    @cache.cached
    def my_id(self, x):
        return id(self)


@pytest.mark.parametrize('class_type', [IamMemoryCached, IamDiskCached])
def test_runtime(class_type):
    r = class_type()
    val = 'koko'
    int0 = datetime.now()
    r.me_takey_long_time(val)
    int1 = datetime.now()
    r.me_takey_long_time(val)
    int2 = datetime.now()
    assert int0 < int1 <= int2
    delta1 = int1 - int0
    delta2 = int2 - int1
    assert delta1 >= SLEEP_DELTA, r
    assert delta2 < delta1, r
    assert delta2 < 0.5 * SLEEP_DELTA, r


@contextlib.contextmanager
def _close_cache(backend):
    """This avoids the tmp dir trying to rm still open files with the disk backend"""
    yield
    try:
        backend._cache.close()
    except AttributeError:
        pass


def test_region_api():
    key = 'mykey'
    with tempfile.TemporaryDirectory() as tmpdir:
        backends = [cache.MemoryRegion(100),
                    cache.DiskRegion(path=os.path.join(tmpdir, str(uuid4())),
                                     max_size=1024 ** 2, persistent=False)]
        for backend in backends:
            with _close_cache(backend):
                assert backend.get(key) == (False, None)
                backend.set(key, 1)
                assert backend.get(key) == (True, 1)
                # second set is ignored
                backend.set(key, 2)
                assert backend.get(key) == (True, 1)
                backend.clear()
                assert backend.get(key) == (False, None)


def test_memory_region_safety():

    op = NumpyMatrixOperator(np.eye(1))
    rhs = op.range.make_array(np.array([1]))
    m = StationaryModel(op, rhs)
    m.enable_caching('memory')

    U = m.solve()
    del U[:]
    U = m.solve()
    assert len(U) == 1
    del U[:]
    U = m.solve()
    assert len(U) == 1


if __name__ == "__main__":
    runmodule(filename=__file__)
