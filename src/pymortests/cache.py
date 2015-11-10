# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import time
import os
from uuid import uuid4
from datetime import datetime
from tempfile import gettempdir

from pymor.core import cache
from pymortests.base import TestInterface, runmodule

SLEEP_SECONDS = 0.2


class IamMemoryCached(cache.CacheableInterface):

    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(SLEEP_SECONDS)
        return arg


class IamDiskCached(cache.CacheableInterface):

    def __init__(self):
        self.cache_region = 'disk'

    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(SLEEP_SECONDS)
        return arg


class IamLimitedCached(cache.CacheableInterface):

    def __init__(self, region='disk'):
        self.cache_region = region

    @cache.cached
    def me_takey_no_time(self, arg):
        return int(arg)


class IWillBeCopied(cache.CacheableInterface):

    def __init__(self):
        super(IWillBeCopied, self).__init__()

    @cache.cached
    def my_id(self, x):
        return id(self)


class TestCache(TestInterface):

    def test_runtime(self):
        for Class in [IamMemoryCached, IamDiskCached]:
            r = Class()
            for val in ['koko', 'koko', 'other']:
                int0 = datetime.now()
                r.me_takey_long_time(val)
                int1 = datetime.now()
                self.logger.info(int1 - int0)

    # def test_limit(self):
    #     for c in [IamLimitedCached('memory'),
    #               IamLimitedCached('disk')]:
    #         for i in range(25):
    #             c._cache_region.backend.print_limit()
    #             c.me_takey_no_time(i)
    #             c._cache_region.backend.print_limit()

    # This test will now fail since x and y will have the same sid.
    # def test_copy(self):
    #     from copy import copy
    #     x = IWillBeCopied()
    #     x_id = x.my_id(1)
    #     y = copy(x)
    #     y_id = y.my_id(1)
    #     self.assertNotEqual(x_id, y_id)

    def test_region_api(self):
        tempdir = gettempdir()
        backends = [cache.MemoryRegion(100), cache.SQLiteRegion(path=os.path.join(tempdir, str(uuid4())),
                                                                max_size=1024 ** 2, persistent=False)]
        for backend in backends:
            assert not backend.get('mykey')[0]
            backend.set('mykey', 1)
            assert backend.get('mykey') == (True, 1)


if __name__ == "__main__":
    runmodule(filename=__file__)
