from __future__ import absolute_import, division, print_function

import tempfile

from pymor import *
from pymor.core.interfaces import BasicInterface
from pymortests.base import TestBase, runmodule
from pymor.core import (dump, load)


class PickleMe(TestBase):
    
    def testDump(self):
        
        for Type in BasicInterface.implementors(True):
            if Type.has_interface_name():
                continue
            try:
                inst = Type()
            except ValueError:
                #no default init -> just test next type
                continue
            dump(inst, tempfile.TemporaryFile())

if __name__ == "__main__":
    runmodule(name='pymortests.core.pickling')