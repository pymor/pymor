import nose
import time
from datetime import datetime

from pymor.core import cache
from pymor.tests.base import TestBase

SLEEP_SECONDS = 2

class IamMemoryCached(cache.Cachable):
   
    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(SLEEP_SECONDS)
        return arg

class IamDiskCached(cache.Cachable):
    
    def __init__(self):
        super(IamDiskCached, self).__init__(config=cache.DEFAULT_DISK_CONFIG)

    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(SLEEP_SECONDS)
        return arg

class CacheTest(TestBase):
       
    def test_runtime(self):
        for Class in [IamMemoryCached, IamDiskCached]:
            r = Class()
            int0 = datetime.now()
            r.me_takey_long_time('koko')
            int1 = datetime.now()
            self.logger.info(int1-int0)
            r.me_takey_long_time('koko')
            int2 = datetime.now()
            self.logger.info(int2-int1)
            r.me_takey_long_time('other')
            int4 = datetime.now()
            self.logger.warning(int4-int3)

if __name__ == "__main__":
    nose.core.runmodule(name='__main__')
