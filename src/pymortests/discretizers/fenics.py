import pytest

from pymor.analyticalproblems.elliptic import StationaryProblem
from pymortests.base import runmodule
from pymortests.base import skip_if_missing


@skip_if_missing('FENICS')
def test_discretize(analytical_problem):
    from pymor.discretizers.fenics.cg import discretize_stationary_cg
    if not isinstance(analytical_problem, StationaryProblem):
        pytest.skip('fenics does not provide InstationaryProblem discretization')
    discretize_stationary_cg(analytical_problem)


if __name__ == "__main__":
    runmodule(filename=__file__)
