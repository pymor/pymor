from __future__ import absolute_import, division, print_function

from pymor.core import defaults
from pymortests.base import TestBase, runmodule


class DefaultsTest(TestBase):

    def testStr(self):
        rep = str(defaults)
        self.assertGreater(len(rep), 0)

if __name__ == "__main__":
    runmodule(name='pymortests.core.defaults')
