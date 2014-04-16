# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor import defaults
from pymortests.base import TestInterface, runmodule


class TestDefaults(TestInterface):

    def testStr(self):
        rep = str(defaults)
        self.assertGreater(len(rep), 0)

if __name__ == "__main__":
    runmodule(filename=__file__)
