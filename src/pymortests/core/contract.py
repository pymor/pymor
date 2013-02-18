from __future__ import absolute_import, division, print_function
from nose.tools import raises
import mock

from pymor.core import exceptions
from pymor.core.interfaces import ( contract, abstractmethod, abstractstaticmethod, 
                                   abstractclassmethod)
from pymor.core import timing
from pymor.core.exceptions import ContractNotRespected
from pymor.grids import AllDirichletBoundaryInfo as ADIA
from pymor.playground.boundaryinfos.oned import AllDirichletBoundaryInfo as ADIB
import pymor.grids.boundaryinfos
import pymor.playground.boundaryinfos.oned
from pymortests.base import TestBase, runmodule
from pymortests.core.dummies import *


class ContractTest(TestBase):
    
    @raises(ContractNotRespected)
    def testContractFail(self):
        AverageImplementer().whisper('Wheee\n', -2)

    def testContractSuccess(self):
        AverageImplementer().shout('Wheee\n', 6)
        
    def testNaming(self):
        imp = BoringTestClass()
        def _combo(dirichletA, dirichletB):
            self.assertTrue(imp.dirichletTest(dirichletA, dirichletB))
            with self.assertRaises(ContractNotRespected): 
                imp.dirichletTest(dirichletA, dirichletA)
            with self.assertRaises(ContractNotRespected): 
                imp.dirichletTest(dirichletB, dirichletA)
            with self.assertRaises(ContractNotRespected): 
                imp.dirichletTest(dirichletA, 1)
        grid = mock.Mock()
        dirichletA = pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid)
        dirichletB = pymor.playground.boundaryinfos.oned.AllDirichletBoundaryInfo()
        _combo(dirichletA, dirichletB)
        dirichletA = ADIA(grid)
        dirichletB = ADIB()
        _combo(dirichletA, dirichletB)
        
    def test_custom_contract_types(self):
        inst = BoringTestClass()
        with self.assertRaises(exceptions.ContractNotRespected):
            grid = mock.Mock()
            inst.validate_interface(object(), pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid))
        inst.validate_interface(BoringTestInterface(), pymor.playground.boundaryinfos.oned.AllDirichletBoundaryInfo())
        
if __name__ == "__main__":
    runmodule(name='pymortests.core.contract')
