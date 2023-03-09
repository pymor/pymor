import pytest

from pymor.analyticalproblems.elliptic import StationaryProblem
from pymortests.base import runmodule, skip_if_missing


@skip_if_missing('FENICS')
def test_discretize(analytical_problem):
    from pymor.discretizers.fenics.cg import discretize_stationary_cg
    if not isinstance(analytical_problem, StationaryProblem):
        pytest.skip('fenics does not provide InstationaryProblem discretization')
    try:
        discretize_stationary_cg(analytical_problem)
    except NotImplementedError as e:
        if 'Conversion to UFL' in str(e):
            pytest.skip('Problem contains functions that cannot be converted to UFL expression.')
        raise


if __name__ == '__main__':
    runmodule(filename=__file__)
