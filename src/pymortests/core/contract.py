# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import pytest

try:
    import contracts
    HAVE_CONTRACTS = True
except ImportError:
    HAVE_CONTRACTS = False

if HAVE_CONTRACTS:
    import mock

    from pymortests.base import TestInterface, runmodule
    import pymor.grids.boundaryinfos
    from pymortests.core.dummies import (AllDirichletBoundaryInfo, AverageImplementer, BoringTestClass)
    from pymor.core import exceptions
    from pymor.core.interfaces import (contract,)
    from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo as ADIA
    from pymor.core.exceptions import ContractNotRespected


    class TestContract(TestInterface):

        def test_contract_fail(self):
            with pytest.raises(ContractNotRespected):
                AverageImplementer().whisper('Wheee\n', -2)

        def test_contract_success(self):
            AverageImplementer().shout('Wheee\n', 6)

        def test_naming(self):
            imp = BoringTestClass()

            def _combo(dirichlet_a, dirichlet_b):
                assert imp.dirichletTest(dirichlet_a, dirichlet_b)
                with pytest.raises(ContractNotRespected):
                    imp.dirichletTest(dirichlet_a, dirichlet_a)
                with pytest.raises(ContractNotRespected):
                    imp.dirichletTest(dirichlet_b, dirichlet_a)
                with pytest.raises(ContractNotRespected):
                    imp.dirichletTest(dirichlet_a, 1)
            grid = mock.Mock()
            dirichlet_b = AllDirichletBoundaryInfo()
            dirichlet_a = pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid)
            _combo(dirichlet_a, dirichlet_b)
            dirichlet_a = ADIA(grid)
            _combo(dirichlet_a, dirichlet_b)

        def test_custom_contract_types(self):
            inst = BoringTestClass()
            with pytest.raises(exceptions.ContractNotRespected):
                grid = mock.Mock()
                inst.validate_interface(object(), pymor.grids.boundaryinfos.AllDirichletBoundaryInfo(grid))

        def test_disabled_contracts(self):
            contracts.disable_all()

            @contract
            def disabled(phrase):
                """
                :type phrase: str
                """
                return phrase
            # this should not throw w/ contracts disabled
            disabled(int(8))
            contracts.enable_all()
            # this will still not throw because the disabled value is checked at decoration time only
            disabled(int(8))

            @contract
            def enabled(phrase):
                """
                :type phrase: str
                """
                return phrase
            # a newly decorated function will throw
            with pytest.raises(exceptions.ContractNotRespected):
                enabled(int(8))

if __name__ == "__main__":
    runmodule(filename=__file__)
