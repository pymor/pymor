import nose
import time
from datetime import datetime

from pymor.core import cache
from pymor.tests.base import TestBase

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
    
class IamLimitedMemoryCached(cache.Cachable):
    
    def __init__(self, config=cache.DEFAULT_DISK_CONFIG):
        super(IamLimitedMemoryCached, self).__init__(config=config)

    @cache.cached
    def me_takey_no_time(self, arg):
        return int(arg)
    
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
        c = IamLimitedMemoryCached(cache.SMALL_MEMORY_CONFIG)
        for i in '1234567':
            c.cache_region.backend.print_limit()
            k = c.me_takey_no_time(i)
            c.cache_region.backend.print_limit()
            

if __name__ == "__main__":
    nose.core.runmodule(name='__main__')
