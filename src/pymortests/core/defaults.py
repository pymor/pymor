# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor import defaults
from pymortests.base import TestBase, runmodule


class DefaultsTest(TestBase):

    def testStr(self):
        rep = str(defaults)
        self.assertGreater(len(rep), 0)

if __name__ == "__main__":
    runmodule(name='pymortests.core.defaults')
