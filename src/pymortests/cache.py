# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import time
from datetime import datetime

from pymor.core import cache
from pymortests.base import TestInterface, runmodule

SLEEP_SECONDS = 0.2


class IamMemoryCached(cache.Cachable):

    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(SLEEP_SECONDS)
        return arg


class IamDiskCached(cache.Cachable):

    def __init__(self, ):
        super(IamDiskCached, self).__init__(config=cache.DEFAULT_DISK_CONFIG)

    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(SLEEP_SECONDS)
        return arg


class IamLimitedCached(cache.Cachable):

    def __init__(self, config=cache.DEFAULT_DISK_CONFIG):
        super(IamLimitedCached, self).__init__(config=config)

    @cache.cached
    def me_takey_no_time(self, arg):
        return int(arg)


class IWillBeCopied(cache.Cachable):

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

    def test_limit(self):
        for c in [IamLimitedCached(cache.SMALL_MEMORY_CONFIG),
                  IamLimitedCached(cache.SMALL_DISK_CONFIG)]:
            for i in range(25):
                c.cache_region.backend.print_limit()
                _ = c.me_takey_no_time(i)
                c.cache_region.backend.print_limit()

    def test_copy(self):
        from copy import copy
        x = IWillBeCopied()
        x_id = x.my_id(1)
        y = copy(x)
        y_id = y.my_id(1)
        self.assertNotEqual(x_id, y_id)

    def test_backend_api(self):
        for backend_cls in [cache.LimitedFileBackend, cache.LimitedMemoryBackend, cache.DummyBackend]:
            backend = backend_cls({})
            self.assertEqual(backend.get('mykey'), cache.NO_VALUE)
            backend.set('mykey', 1)
            self.assertEqual(backend.get('mykey'), 1 if backend_cls != cache.DummyBackend else cache.NO_VALUE)
            backend.delete('mykey')
            self.assertEqual(backend.get('mykey'), cache.NO_VALUE)


if __name__ == "__main__":
    runmodule(filename=__file__)
