import nose
import time
from datetime import datetime

from pymor.core import cache
from pymor.tests.base import TestBase


class MyRun(cache.Cachable):
    
    sleep_secs = 2
    
    @cache.cached
    def me_takey_long_time(self, arg):
        time.sleep(self.sleep_secs)
        return arg

class CacheTest(TestBase):
    
    
    def test_runtime(self):
        r = MyRun()
        int0 = datetime.now()
        r.me_takey_long_time('koko')
        int1 = datetime.now()
        self.logger.info(int1-int0)
        r.me_takey_long_time('koko')
        int2 = datetime.now()
        self.logger.info(int2-int1)
#        r.me_takey_long_time('koko')
#        int3 = datetime.now()
#        self.logger.info(int3-int2)
#        r.me_takey_long_time('other')
#        int4 = datetime.now()
#        self.logger.warning(int4-int3)

if __name__ == "__main__":
    nose.core.runmodule(name='__main__')
