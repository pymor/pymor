# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
import tempfile
import time
import os
from uuid import uuid4
from datetime import datetime, timedelta

from pymor.core import cache
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


def test_runtime():
    for Class in [IamMemoryCached, IamDiskCached]:
        r = Class()
        val = 'koko'
        int0 = datetime.now()
        r.me_takey_long_time(val)
        int1 = datetime.now()
        r.me_takey_long_time(val)
        int2 = datetime.now()
        assert int0 < int1 <= int2
        delta1 = int1 - int0
        delta2 = int2 - int1
        assert delta1>= SLEEP_DELTA, r
        assert delta2 < delta1, r
        assert delta2 < 0.5 * SLEEP_DELTA, r


def test_region_api():
    with tempfile.TemporaryDirectory() as tmpdir:
        backends = [cache.MemoryRegion(100), cache.DiskRegion(path=os.path.join(tmpdir, str(uuid4())),
                                                                max_size=1024 ** 2, persistent=False)]
        for backend in backends:
            assert backend.get('mykey') == (False, None)
            backend.set('mykey', 1)
            assert backend.get('mykey') == (True, 1)
            # second set is ignored
            backend.set('mykey', 2)
            assert backend.get('mykey') == (True, 1)


if __name__ == "__main__":
    runmodule(filename=__file__)
