# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import pytest

from pymortests.base import TestInterface, runmodule
from pymortests.core.dummies import (AllDirichletBoundaryInfo, AverageImplementer, BoringTestClass)
import pymor.grids.boundaryinfos
from pymor.core import exceptions
from pymor.core.interfaces import (contract,)
from pymor.grids import AllDirichletBoundaryInfo as ADIA


try:
    import contracts
    import mock

    from pymor.core.exceptions import ContractNotRespected

    class TestContract(TestInterface):

        def testContractFail(self):
            with pytest.raises(ContractNotRespected):
                AverageImplementer().whisper('Wheee\n', -2)

        def testContractSuccess(self):
            AverageImplementer().shout('Wheee\n', 6)

        def testNaming(self):
            imp = BoringTestClass()

            def _combo(dirichletA, dirichletB):
                self.assertTrue(imp.dirichletTest(dirichletA, dirichletB))
                with pytest.raises(ContractNotRespected):
                    imp.dirichletTest(dirichletA, dirichletA)
                with pytest.raises(ContractNotRespected):
                    imp.dirichletTest(dirichletB, dirichletA)
                with pytest.raises(ContractNotRespected):
                    imp.dirichletTest(dirichletA, 1)
            grid = mock.Mock()
            dirichletB = AllDirichletBoundaryInfo()
            dirichletA = pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid)
            _combo(dirichletA, dirichletB)
            dirichletA = ADIA(grid)
            _combo(dirichletA, dirichletB)

        def test_custom_contract_types(self):
            inst = BoringTestClass()
            with pytest.raises(exceptions.ContractNotRespected):
                grid = mock.Mock()
                inst.validate_interface(object(), pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid))

        def test_disabled_contracts(self):
            contracts.disable_all()

            @contract
            def disabled(phrase):
                '''
                :type phrase: str
                '''
                return phrase
            # this should not throw w/ contracts disabled
            disabled(int(8))
            contracts.enable_all()
            # this will still not throw because the disabled value is checked at decoration time only
            disabled(int(8))

            @contract
            def enabled(phrase):
                '''
                :type phrase: str
                '''
                return phrase
            # a newly decorated function will throw
            with pytest.raises(exceptions.ContractNotRespected):
                enabled(int(8))

except ImportError:
    pass


if __name__ == "__main__":
    runmodule(filename=__file__)
