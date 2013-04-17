# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import tempfile
import os

from pymor.core.interfaces import (BasicInterface,)
from pymortests.base import (TestBase, runmodule, SubclassForImplemetorsOf)
from pymor import core


@SubclassForImplemetorsOf(BasicInterface)
class PickleMeInterface(TestBase):

    def testDump(self):
        try:
            obj = self.Type()
            self.assertIsInstance(obj, self.Type)
            if issubclass(self.Type, core.Unpicklable):
                return
        except TypeError as e:
            self.logger.debug('PicklingError: Not testing {} because its init failed: {}'.format(self.Type, str(e)))
            return

        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as dump_file:
            core.dump(obj, dump_file)
            dump_file.close()
            f = open(dump_file.name, 'rb')
            unpickled = core.load(f)
            self.assertEqual(obj.__class__, unpickled.__class__)
            os.unlink(dump_file.name)

# this needs to go into every module that wants to use dynamically generated types, ie. testcases, below the test code
from pymor.core.dynamic import *

if __name__ == "__main__":
    runmodule(name='pymortests.core.pickling')
