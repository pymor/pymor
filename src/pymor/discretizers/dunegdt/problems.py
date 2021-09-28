# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_DUNEGDT:

    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import Function
    from pymor.core.base import classinstancemethod
    from pymor.discretizers.dunegdt.domaindiscretizers.default import discretize_domain_default
    from pymor.discretizers.dunegdt.functions import to_dune_grid_function, DuneInterpolator
    from pymor.parameters.base import ParametricObject


    class StationaryDuneProblem(ParametricObject):
        def __init__(self, grid, boundary_info, rhs=None, diffusion=None, advection=None, reaction=None,
                     dirichlet_data=None, neumann_data=None, robin_data=None, outputs=None, parameter_space=None,
                     name=None, interpolator=None, data_approximation_order=1):

            # create common interpolator to cache discrete function spaces and interpolation points
            if interpolator is None:
                assert int(data_approximation_order) == data_approximation_order and data_approximation_order >= 0
                interpolator = DuneInterpolator(
                        grid,
                        space_type='fv' if data_approximation_order == 0 else \
                                ('cg' if data_approximation_order == 1 else 'dg'),
                        order=data_approximation_order)
            boundary_interpolator = interpolator if data_approximation_order == 1 else DuneInterpolator(grid, 'cg', 1)

            # ensure all data functions are dune compatible
            # - those arising mainly in volume integrals
            diffusion = to_dune_grid_function(diffusion, grid, interpolator, ensure_lincomb=True) \
                    if diffusion is not None else None
            rhs = to_dune_grid_function(rhs, grid, interpolator, ensure_lincomb=True) \
                    if rhs is not None else None
            reaction = to_dune_grid_function(reaction, grid, interpolator, ensure_lincomb=True) \
                    if reaction is not None else None
            # - those arising in intersection integrals as well as Dirichlet data
            advection = to_dune_grid_function(advection, grid, interpolator, ensure_lincomb=True) \
                    if advection is not None else None
            dirichlet_data = to_dune_grid_function(dirichlet_data, grid, interpolator, ensure_lincomb=True) \
                    if dirichlet_data is not None else None
            neumann_data = to_dune_grid_function(neumann_data, grid, interpolator, ensure_lincomb=True) \
                    if neumann_data is not None else None
            # - Robin data
            if robin_data is not None:
                assert isinstance(robin_data, tuple) and len(robin_data) == 2
                robin_data = tuple(to_dune_grid_function(func, grid, interpolator, ensure_lincomb=True)
                                   for func in robin_data)
            # - outputs
            if outputs is not None:
                assert isinstance(outputs, (tuple, list)) and all(len(o) == 2 for o in outputs) \
                       and all(o[0] in ('l2', 'l2_boundary') for o in outputs)
                for output in outputs:
                    output[1] = to_dune_grid_function(
                            output[1],
                            grid,
                            interpolator if output[0] == 'l2' else boundary_interpolator)

            name = name or 'StationaryDuneProblem'
            self.__auto_init(locals())

        @classinstancemethod
        def from_pymor(cls, analytical_problem,
                       data_approximation_order,
                       diameter=None, domain_discretizer=None,
                       grid_type=None, grid=None, boundary_info=None):
            # some checks
            p = analytical_problem
            assert isinstance(p, StationaryProblem)
            if not (p.nonlinear_advection
                    == p.nonlinear_advection_derivative
                    == p.nonlinear_reaction
                    == p.nonlinear_reaction_derivative
                    is None):
                raise NotImplementedError

            assert grid is None or boundary_info is not None
            assert boundary_info is None or grid is not None
            assert grid is None or domain_discretizer is None
            assert grid_type is None or grid is None

            # build grid if required
            if grid is None:
                domain_discretizer = domain_discretizer or discretize_domain_default
                if grid_type:
                    domain_discretizer = partial(domain_discretizer, grid_type=grid_type)
                if diameter is None:
                    grid, boundary_info = domain_discretizer(p.domain)
                else:
                    grid, boundary_info = domain_discretizer(p.domain, diameter=diameter)

            return StationaryDuneProblem(grid, boundary_info, p.rhs, p.diffusion, p.advection, p.reaction,
                    p.dirichlet_data, p.neumann_data, p.robin_data, p.outputs, parameter_space=p.parameter_space,
                    name=p.name)
