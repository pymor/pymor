from __future__ import absolute_import, division, print_function

import tempfile
import os

from pymor import *
from pymor.core import *
from pymor.grids import *
from pymor.discreteoperators import *
from pymor.discretizations import *
from pymor.discretizers import *
from pymor.domaindescriptions import *
from pymor.domaindiscretizers import *
from pymor.functions import *
from pymor.reductors import *
from pymor.tools import *

from pymor.core.interfaces import (BasicInterface,)
from pymortests.base import (TestBase, runmodule, SubclassForImplemetorsOf)
from pymor.core import (dump, load)
import pymor
    
@SubclassForImplemetorsOf(BasicInterface)
class PickleMeInterface(TestBase):
    
    def testDump(self):
        try:
            obj = self.Type
        except (ValueError, TypeError):
            #no default init -> just test next type
            return
        self.logger.critical('Testing ' +str(obj))
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as dump_file:
            obj.some_attribute = 4
            dump(obj, dump_file)
            dump_file.close()
            f = open(dump_file.name, 'rb')
            unpickled = load(f)
            
            os.unlink(dump_file.name)
        dump(obj, tempfile.TemporaryFile())

#this needs to go into every module that wants to use dynamically generated types, ie. testcases, below the test code
from pymor.core.dynamic import *
if __name__ == "__main__":
    runmodule(name='pymortests.core.pickling')
    for fu,du in pymor.__dict__.items():        
        if 'Pickle' in fu:
            print('{} -- {}'.format(fu, du.Type))
            