import nose
import time
from datetime import datetime

from pymor.core.cache import cached
from pymor.tests.base import TestBase

class CacheTest(TestBase):
    
    def test_runtime(self):
        sleep_secs = 0
        @cached
        def _me_takey_long_time(arg):
            time.sleep(sleep_secs)
            return arg
        int0 = datetime.now()
        _me_takey_long_time('koko')
        int1 = datetime.now()
        self.logger.error(int1-int0)
        _me_takey_long_time('koko')
        int2 = datetime.now()
        self.logger.critical(int2-int1)
        _me_takey_long_time('koko')
        int3 = datetime.now()
        self.logger.info(int3-int2)
        _me_takey_long_time('other')
        int4 = datetime.now()
        self.logger.warning(int4-int3)

if __name__ == "__main__":
    nose.core.runmodule(name='__main__')
