from __future__ import absolute_import, division, print_function
import time
from datetime import datetime

from pymor.core import cache
from pymortests.base import TestBase, runmodule

SLEEP_SECONDS = 1

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


class CacheTest(TestBase):
       
    def test_runtime(self):
        for Class in [IamMemoryCached, IamDiskCached]:
            r = Class()
            for val in ['koko', 'koko', 'other']:                
                int0 = datetime.now()
                r.me_takey_long_time(val)
                int1 = datetime.now()
                self.logger.info(int1-int0)
            
    def test_limit(self):
        for c in [IamLimitedCached(cache.SMALL_MEMORY_CONFIG), 
                  IamLimitedCached(cache.SMALL_DISK_CONFIG)]:
            for i in range(25):
                c.cache_region.backend.print_limit()
                k = c.me_takey_no_time(i)
                c.cache_region.backend.print_limit()

    def test_copy(self):
        from copy import copy
        x = IWillBeCopied()
        x_id = x.my_id(1)
        y = copy(x)
        y_id = y.my_id(1)
        self.assertNotEqual(x_id, y_id)

if __name__ == "__main__":
    runmodule(name='pymortests.cache')
