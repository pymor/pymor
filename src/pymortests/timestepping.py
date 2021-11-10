# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper, ExplicitEulerTimeStepper, ExplicitRungeKuttaTimeStepper
from pymor.analyticalproblems.domaindescriptions import LineDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.discretizers.builtin.fv import discretize_instationary_fv
from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.discretizers.builtin.grids.oned import OnedGrid

from pymortests.base import runmodule

problem = InstationaryProblem(
    StationaryProblem(LineDomain(), advection=ConstantFunction(dim_domain=1, value=np.array([1,]))),
    ExpressionFunction(dim_domain=1, expression='1.*(0.1 <= x[0])*(x[0] <= 0.2)'),
    T=1,
    name='linear_transport'
)

grid = OnedGrid(problem.stationary_part.domain.domain, 10, identify_left_right=True)

fom, _fom_data = discretize_instationary_fv(
    problem,
    grid=grid,
    boundary_info=EmptyBoundaryInfo(grid),
    num_flux='upwind',
    time_stepper=ExplicitEulerTimeStepper(nt=grid.size(0), interpolation='P0')
)


def test_ExplicitEulerTimeStepper():
    fom_ = fom.with_(time_stepper=ExplicitEulerTimeStepper(nt=grid.size(0)))
    U = fom_.solve()
    assert len(U) == grid.size(0) + 1
    assert almost_equal(U[0], U[-1])


RK_expected_results = {
    'explicit_euler': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'RK1': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'heun2': [0.0, 0.248046875, 0.0, 0.21484375, 0.0, 0.1611328125, 0.0, 0.1611328125, 0.0, 0.21484375],
    'midpoint': [0.0, 0.248046875, 0.0, 0.21484375, 0.0, 0.1611328125, 0.0, 0.1611328125, 0.0, 0.21484375],
    'ralston': [-4.163336342344337e-17, 0.24804687499999992, -6.938893903907228e-18, 0.21484375, 0.0, 0.1611328125, -2.0816681711721685e-17, 0.16113281249999997, -5.551115123125783e-17, 0.21484374999999994],
    'RK2': [0.0, 0.248046875, 0.0, 0.21484375, 0.0, 0.1611328125, 0.0, 0.1611328125, 0.0, 0.21484375],
    'simpson': [0.12750037310115328, 0.12502245883715218, 0.1135725864324544, 0.09631566580297718, 0.08010296533387526, 0.07209758063747906, 0.07466647138393538, 0.08693132835124218, 0.10414913620467746, 0.11964143391505357],
    'heun3': [0.12750037310115325, 0.12502245883715218, 0.1135725864324544, 0.09631566580297721, 0.08010296533387527, 0.07209758063747904, 0.07466647138393535, 0.08693132835124216, 0.10414913620467742, 0.11964143391505358],
    'RK3': [0.12750037310115328, 0.12502245883715218, 0.1135725864324544, 0.09631566580297718, 0.08010296533387526,
        0.07209758063747906, 0.07466647138393538, 0.08693132835124218, 0.10414913620467746, 0.11964143391505357],
    '3/8': [0.1291366358395182, 0.1271438146630103, 0.11501806234690207, 0.09731030053861438, 0.08047406795019868, 0.07086134225986314, 0.0723735221596757, 0.08468566319313724, 0.1029892441598012, 0.12000734688927919],
    'RK4': [0.1291366358395182, 0.12714381466301025, 0.11501806234690208, 0.09731030053861438, 0.08047406795019867, 0.07086134225986314, 0.07237352215967573, 0.08468566319313726, 0.1029892441598012, 0.12000734688927922],
}


def test_ExplicitEulerTimeStepper():
    for method in ExplicitRungeKuttaTimeStepper.available_RK_methods:
        fom_ = fom.with_(time_stepper=ExplicitRungeKuttaTimeStepper(method=method, nt=grid.size(0)))
        U = fom_.solve()
        assert len(U) == grid.size(0) + 1
        assert method in RK_expected_results
        U_desired = fom_.solution_space.from_numpy(np.array(RK_expected_results[method]))
        assert almost_equal(U[-1], U_desired)


def test_ImplicitEulerTimeStepper():
    fom_ = fom.with_(time_stepper=ImplicitEulerTimeStepper(nt=grid.size(0)))
    U = fom_.solve()
    assert len(U) == grid.size(0) + 1
    U_desired = fom_.solution_space.from_numpy(np.array([0.10589990845086997, 0.09859686010245765, 0.09176013215968842, 0.08828032877798837, 0.08946896633499435, 0.09462565960828838, 0.10160967696802872, 0.10785309463124143, 0.11125284854251424, 0.11065252442392833]))
    assert almost_equal(U[-1], U_desired)

if __name__ == "__main__":
    runmodule(filename=__file__)
