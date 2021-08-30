from pymor.core.config import config, is_jupyter


if config.HAVE_DUNEGDT:
    import numpy as np
    from functools import partial

    from dune.xt.grid import (
            ApplyOnBoundaryIntersections,
            ApplyOnCustomBoundaryIntersections,
            Dim,
            DirichletBoundary,
            NeumannBoundary,
            RobinBoundary,
            Walker,
            )
    from dune.xt.functions import divergence, GridFunction as GF
    from dune.xt.la import Istl
    from dune.gdt import (
            ContinuousLagrangeSpace,
            DirichletConstraints,
            DiscontinuousLagrangeSpace,
            DiscreteFunction,
            LocalElementIntegralBilinearForm,
            LocalElementIntegralFunctional,
            LocalElementProductIntegrand,
            LocalIntersectionIntegralBilinearForm,
            LocalIntersectionIntegralFunctional,
            LocalIntersectionNormalComponentProductIntegrand,
            LocalIntersectionProductIntegrand,
            LocalLaplaceIntegrand,
            LocalLinearAdvectionIntegrand,
            MatrixOperator,
            VectorFunctional,
            boundary_interpolation,
            make_element_sparsity_pattern,
            )

    from pymor.algorithms.timestepping import ExplicitEulerTimeStepper, ImplicitEulerTimeStepper
    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.instationary import InstationaryProblem
    from pymor.analyticalproblems.functions import Function, ConstantFunction, LincombFunction
    from pymor.bindings.dunegdt import (
            DuneGDT1dasNumpyVisualizer,
            DuneGDT1dMatplotlibVisualizer,
            DuneGDTK3dVisualizer,
            DuneGDTParaviewVisualizer,
            DuneXTMatrixOperator,
            DuneXTVector,
            DuneXTVectorSpace,
            )
    from pymor.core.base import ImmutableObject
    from pymor.discretizers.dunegdt.domaindiscretizers.default import discretize_domain_default
    from pymor.models.basic import InstationaryModel, StationaryModel
    from pymor.operators.constructions import ConstantOperator, LincombOperator, VectorArrayOperator
    from pymor.tools.floatcmp import float_cmp


    def discretize_stationary_cg(analytical_problem, diameter=None, domain_discretizer=None,
                                 grid_type=None, grid=None, boundary_info=None,
                                 order=1, data_approximation_order=2, la_backend=Istl(),
                                 advection_in_divergence_form=True, mu_energy_product=None):
        """Discretizes a |StationaryProblem| with dune-gdt using continuous Lagrange finite elements.

        Note: all data functions are replaced by their respective non-conforming interpolations. This allows to simply
              use pyMORs data |Function|s at the expense of one DoF vector for each data function during discretization.

        Note: non-trivial Dirichlet data is treated via shifting. The resulting solution is thus in H^1_0 and the shift
              is added upon visualization or output computation.

        TODO: check if all products still make sense!

        Parameters
        ----------
        analytical_problem
            The |StationaryProblem| to discretize.
        diameter
            If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
        domain_discretizer
            Discretizer to be used for discretizing the analytical domain. This has
            to be a function `domain_discretizer(domain_description, diameter, ...)`.
            If `None`, |discretize_domain_default| is used.
        grid_type
            If not `None`, this parameter is forwarded to `domain_discretizer` to specify
            the type of the generated |Grid|.
        grid
            Instead of using a domain discretizer, the |Grid| can also be passed directly
            using this parameter.
        boundary_info
            A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
            Must be provided if `grid` is specified.
        order
            Order of the Finite Element space.
        data_approximation_order
            Polynomial order (on each grid element) for the interpolation of the data functions.
        la_backend
            Tag to determine which linear algebra backend from dune-xt is used.
        advection_in_divergence_form
            If true, treats linear advection as advertised in StationaryProblem (i.e. :math:`∇ ⋅ (v u)`), else as in
            :math:`v ⋅∇ u` (where :math:`v` denotes the vector field).

        Returns
        -------
        m
            The |Model| that has been generated.
        data
            Dictionary with the following entries:

                :grid:                  The generated grid from dune.xt.grid.
                :boundary_info:         The generated boundary info from dune.xt.grid.
                :space:                 The generated approximation space from dune.gdt.
                :dirichlet_shift:       A |VectorArray| respresenting the Dirichlet shift.
                :unshifted_visualizer:  A visualizer which does not add the dirichlet_shift.
        """

        assert isinstance(analytical_problem, StationaryProblem)
        assert grid is None or boundary_info is not None
        assert boundary_info is None or grid is not None
        assert grid is None or domain_discretizer is None
        assert grid_type is None or grid is None

        p = analytical_problem
        d = p.domain.dim

        assert p.dirichlet_data is None or not p.dirichlet_data.parametric

        if not (p.nonlinear_advection
                == p.nonlinear_advection_derivative
                == p.nonlinear_reaction
                == p.nonlinear_reaction_derivative
                is None):
            raise NotImplementedError

        if grid is None:
            domain_discretizer = domain_discretizer or discretize_domain_default
            if grid_type:
                domain_discretizer = partial(domain_discretizer, grid_type=grid_type)
            if diameter is None:
                grid, boundary_info = domain_discretizer(p.domain)
            else:
                grid, boundary_info = domain_discretizer(p.domain, diameter=diameter)

        # prepare to interpolate data functions
        interpolation_space = {'scalar': {}, 'vector': {}}
        interpolation_points = {'scalar': {}, 'vector': {}}

        def interpolate_single(func, pol_order=data_approximation_order):
            assert isinstance(func, Function)
            if func.shape_range in ((), (1,)):
                if not pol_order in interpolation_space['scalar']:
                    interpolation_space['scalar'][pol_order] = DiscontinuousLagrangeSpace(
                            grid, order=pol_order, dim_range=Dim(1))
                if not pol_order in interpolation_points['scalar']:
                    interpolation_points['scalar'][pol_order] = interpolation_space['scalar'][pol_order].interpolation_points()
                df = DiscreteFunction(interpolation_space['scalar'][pol_order], la_backend)
                np_view = np.array(df.dofs.vector, copy=False)
                np_view[:] = func.evaluate(interpolation_points['scalar'][pol_order])[:].ravel()
                return df
            elif func.shape_range == (d,):
                if not pol_order in interpolation_space['vector']:
                    interpolation_space['vector'][pol_order] = DiscontinuousLagrangeSpace(
                            grid, order=pol_order, dim_range=Dim(d))
                if not pol_order in interpolation_points['vector']:
                    interpolation_points['vector'][pol_order] = interpolation_space['vector'][pol_order].interpolation_points()
                df = DiscreteFunction(interpolation_space['vector'][pol_order], la_backend)
                np_view = np.array(df.dofs.vector, copy=False)
                np_view[:] = func.evaluate(interpolation_points['vector'][pol_order])[:].ravel()
                return df
            else:
                raise NotImplementedError(f'I do not know how to interpolate a {func.shape_range}d function!')


        def interpolate(func, pol_order=data_approximation_order):
            if isinstance(func, LincombFunction):
                return [interpolate_single(ff, pol_order) for ff in func.functions], func.coefficients
            elif not isinstance(func, Function):
                func = ConstantFunction(value_array=func, dim_domain=d)
            return [interpolate_single(func, pol_order)], [1]

        # preparations for the actual discretization
        space = ContinuousLagrangeSpace(grid, order=order, dim_range=Dim(1))
        sparsity_pattern = make_element_sparsity_pattern(space)
        constrained_lhs_ops = []
        constrained_lhs_coeffs = []
        unconstrained_lhs_ops = []
        unconstrained_lhs_coeffs = []
        if mu_energy_product is not None:
            mu_energy_product = p.parameters.parse(mu_energy_product)
            energy_product_ops = []
            energy_product_coeffs = []
        rhs_ops = []
        rhs_coeffs = []

        # diffusion part
        def make_diffusion_operator(func):
            op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
            op += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GF(grid, func, (Dim(d), Dim(d)))))
            return op

        if p.diffusion:
            diffusion_funcs, diffusion_coeffs = interpolate(p.diffusion)
            constrained_lhs_ops += [make_diffusion_operator(func) for func in diffusion_funcs]
            constrained_lhs_coeffs += list(diffusion_coeffs)
            if mu_energy_product:
                energy_product_ops += [make_diffusion_operator(func) for func in diffusion_funcs]
                energy_product_coeffs += list(diffusion_coeffs)

        # reaction part
        def make_weighted_l2_operator(func):
             op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
             op += LocalElementIntegralBilinearForm(LocalElementProductIntegrand(GF(grid, func)))
             return op

        if p.reaction:
            reaction_funcs, reaction_coeffs = interpolate(p.reaction)
            constrained_lhs_ops += [make_weighted_l2_operator(func) for func in reaction_funcs]
            constrained_lhs_coeffs += list(reaction_coeffs)
            if mu_energy_product:
                energy_product_ops += [make_weighted_l2_operator(func) for func in reaction_funcs]
                energy_product_coeffs += list(reaction_coeffs)

        # advection part
        if p.advection:
            def make_advection_operator(func):
                op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                op += LocalElementIntegralBilinearForm(LocalLinearAdvectionIntegrand(GF(grid, func),
                                                                                     advection_in_divergence_form))

                if p.diffusion and advection_in_divergence_form: # to ensure Neumann boundary values
                    op += (LocalIntersectionIntegralBilinearForm(
                             LocalIntersectionNormalComponentProductIntegrand(GF(grid, func))), {},
                           ApplyOnCustomBoundaryIntersections(grid, boundary_info, NeumannBoundary()))

                return op

            advection_funcs, advection_coeffs = interpolate(p.advection)
            constrained_lhs_ops += [make_advection_operator(func) for func in advection_funcs]
            constrained_lhs_coeffs += list(advection_coeffs)

            if mu_energy_product:
                energy_product_ops += [make_weighted_l2_operator(divergence(func)) for func in advection_funcs]
                energy_product_coeffs += [-0.5*coeff for coeff in advection_coeffs]

        # robin boundaries
        if p.robin_data:
            assert isinstance(p.robin_data, tuple) and len(p.robin_data) == 2
            robin_parameter_funcs, robin_parameter_coeffs = interpolate(p.robin_data[0])
            robin_boundary_values_funcs, robin_boundary_values_coeffs = interpolate(p.robin_data[1])

            # contributions to the left hand side
            def make_weighted_l2_robin_boundary_operator(func):
                op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                op += (LocalIntersectionIntegralBilinearForm(LocalIntersectionProductIntegrand(GF(grid, func))), {},
                       ApplyOnCustomBoundaryIntersections(grid, boundary_info, RobinBoundary()))
                return op

            unconstrained_lhs_ops += [make_weighted_l2_robin_boundary_operator(func) for func in robin_parameter_funcs]
            unconstrained_lhs_coeffs += list(robin_parameter_coeffs)

            # contributions to the right hand side
            def make_weighted_l2_robin_boundary_functional(r_param_func, r_bv_func):
                op = VectorFunctional(grid, space, la_backend)
                op += (LocalIntersectionIntegralFunctional(
                            LocalIntersectionProductIntegrand(GF(grid, r_param_func)).with_ansatz(r_bv_func)), {},
                       ApplyOnCustomBoundaryIntersections(grid, boundary_info, RobinBoundary()))
                return op

            for r_param_func, r_param_coeff in zip(robin_parameter_funcs, robin_parameter_coeffs):
                for r_bv_func, r_bv_coeff in zip(robin_boundary_values_funcs, robin_boundary_values_coeffs):
                    rhs_ops += [make_weighted_l2_robin_boundary_functional(r_param_func, r_bv_func)]
                    rhs_coeffs += [r_param_coeff*r_bv_coeff]

        # source contribution
        if p.rhs:
            def make_l2_functional(func):
                op = VectorFunctional(grid, space, la_backend)
                op += LocalElementIntegralFunctional(
                        LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(GF(grid, func)))
                return op

            source_funcs, source_coeffs = interpolate(p.rhs)
            rhs_ops += [make_l2_functional(func) for func in source_funcs]
            rhs_coeffs += list(source_coeffs)

        # Neumann boundaries
        if p.neumann_data:
            def make_l2_neumann_boundary_functional(func):
                op = VectorFunctional(grid, space, la_backend)
                op += (LocalIntersectionIntegralFunctional(
                            LocalIntersectionProductIntegrand(GF(grid, -1)).with_ansatz(GF(grid, func))), {},
                       ApplyOnCustomBoundaryIntersections(grid, boundary_info, NeumannBoundary()))
                return op

            neumann_data_funcs, neumann_data_coeffs = interpolate(p.neumann_data)
            rhs_ops += [make_l2_neumann_boundary_functional(func) for func in neumann_data_funcs]
            rhs_coeffs += list(neumann_data_coeffs)

        # Dirichlet boundaries will be handled further below ...

        # products
        l2_product = make_weighted_l2_operator(1)
        h1_semi_product = make_diffusion_operator(1)

        # output functionals
        outputs = []
        if p.outputs:
            if any(v[0] not in ('l2', 'l2_boundary') for v in p.outputs):
                raise NotImplementedError(f'I do not know how to discretize a {v[0]} output!')
            for output_type, output_data in p.outputs:
                if output_type == 'l2':
                    output_data = interpolate_single(output_data)
                    op = VectorFunctional(grid, space, la_backend)
                    op += LocalElementIntegralFunctional(LocalElementProductIntegrand(grid).with_ansatz(output_data))
                    outputs.append(op)
                elif output_type == 'l2_boundary':
                    output_data = interpolate_single(output_data,
                            pol_order=data_approximation_order if data_approximation_order > 0 else 1)
                    op = VectorFunctional(grid, space, la_backend)
                    op += (LocalIntersectionIntegralFunctional(
                            LocalIntersectionProductIntegrand(GF(grid, 1)).with_ansatz(GF(grid, output_data))), {},
                            ApplyOnBoundaryIntersections(grid))
                    outputs.append(op)
                else:
                    raise NotImplementedError(f'I do not know how to discretize a {v[0]} output!')

        # assemble all of the above in one grid walk
        walker = Walker(grid)
        dirichlet_constraints = DirichletConstraints(boundary_info, space)
        walker.append(dirichlet_constraints)
        for op in constrained_lhs_ops:
            walker.append(op)
        for op in unconstrained_lhs_ops:
            walker.append(op)
        for op in rhs_ops:
            walker.append(op)
        walker.append(l2_product)
        walker.append(h1_semi_product)
        if mu_energy_product:
            for op in energy_product_ops:
                walker.append(op)
        for op in outputs:
            walker.append(op)
        walker.walk(thread_parallel=False) # support not stable/enabled yet

        # extract vectors from functionals
        rhs_ops = [op.vector for op in rhs_ops]

        # compute the Dirichlet shift before constraining
        if p.dirichlet_data:
            # we first require an interpolation of first order
            dirichlet_data = interpolate_single(p.dirichlet_data, pol_order=1)
            # second, we restrict this interpolation to the Dirichlet boundary
            dirichlet_data = boundary_interpolation(GF(grid, dirichlet_data), space, boundary_info, DirichletBoundary())
            # third, we only do something if dirichlet_data != 0
            trivial_dirichlet_data = float_cmp(dirichlet_data.dofs.vector.sup_norm(), 0.)
            if not trivial_dirichlet_data:
                for op, coeff in zip(constrained_lhs_ops, constrained_lhs_coeffs):
                    rhs_ops += [op.apply(dirichlet_data.dofs.vector),]
                    rhs_coeffs += [-1*coeff]
        else:
            trivial_dirichlet_data = True

        # prepare additional products
        # - in H^1
        if mu_energy_product:
            energy_product = MatrixOperator(
                    grid, space, space,
                    matrix=LincombOperator(
                        operators=[DuneXTMatrixOperator(op.matrix.copy()) for op in energy_product_ops],
                        coefficients=energy_product_coeffs).assemble(mu=mu_energy_product).matrix)
        # - in H^1_0
        l2_0_product = MatrixOperator(grid, space, space, l2_product.matrix.copy()) # using operators here just for
        h1_0_semi_product = MatrixOperator(grid, space, space, h1_semi_product.matrix.copy()) # unified handling below
        if mu_energy_product:
            energy_product_0 = MatrixOperator(grid, space, space, energy_product.matrix.copy())

        # apply the Dirichlet constraints
        for op in constrained_lhs_ops:
            dirichlet_constraints.apply(op.matrix, only_clear=True, ensure_symmetry=True)
        for vec in rhs_ops:
            dirichlet_constraints.apply(vec) # sets to zero
        dirichlet_constraints.apply(l2_0_product.matrix, ensure_symmetry=True)
        dirichlet_constraints.apply(h1_0_semi_product.matrix, ensure_symmetry=True)
        if mu_energy_product:
            dirichlet_constraints.apply(energy_product_0.matrix, ensure_symmetry=True)

        # create a matrix to hold the unit rows/cols corresponding to Dirichlet DoFs
        op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
        dirichlet_constraints.apply(op.matrix)
        lhs_ops = [op] + constrained_lhs_ops + unconstrained_lhs_ops
        lhs_coeffs = [1.] + constrained_lhs_coeffs + unconstrained_lhs_coeffs

        # wrap everything as pyMOR operators:
        # - lhs
        lhs_ops = [DuneXTMatrixOperator(op.matrix) for op in lhs_ops]
        L = LincombOperator(operators=lhs_ops, coefficients=lhs_coeffs, name='ellipticOperator')

        # - rhs, clean up beforehand
        rhs_ops_ = []
        rhs_coeffs_ = []
        for vec, coeff in zip(rhs_ops, rhs_coeffs):
            if not float_cmp(vec.sup_norm(), 0.):
                rhs_ops_ += [VectorArrayOperator(lhs_ops[0].range.make_array([DuneXTVector(vec)])),]
                rhs_coeffs_ += [coeff,]
        F = LincombOperator(operators=rhs_ops_, coefficients=rhs_coeffs_, name='rhsOperator')
        del rhs_ops, rhs_coeffs

        # - products
        products = {}
        #   * in H^1
        products.update({
            'l2': DuneXTMatrixOperator(l2_product.matrix),
            'h1_semi': DuneXTMatrixOperator(h1_semi_product.matrix),
            'h1': (DuneXTMatrixOperator(l2_product.matrix)
                   + DuneXTMatrixOperator(h1_semi_product.matrix)).assemble(),
        })
        if mu_energy_product:
            products['energy'] = DuneXTMatrixOperator(energy_product.matrix)
        #   * in H^1_0
        products.update({
            'l2_0': DuneXTMatrixOperator(l2_0_product.matrix),
            'h1_0_semi': DuneXTMatrixOperator(h1_0_semi_product.matrix),
            'h1_0': (DuneXTMatrixOperator(l2_0_product.matrix)
                     + DuneXTMatrixOperator(h1_0_semi_product.matrix)).assemble(),
        })
        if mu_energy_product:
            products['energy_0'] = DuneXTMatrixOperator(energy_product_0.matrix)
        if trivial_dirichlet_data:
            dirichlet_data = lhs_ops[0].source.zeros(1)
        else:
            dirichlet_data = lhs_ops[0].source.make_array([DuneXTVector(dirichlet_data.dofs.vector),])

        # - outputs, shift if required
        outputs = [VectorArrayOperator(lhs_ops[0].source.make_array([DuneXTVector(op.vector)]), adjoint=True)
                   for op in outputs]
        if not trivial_dirichlet_data:
            shifted_outputs = []
            for func in outputs:
                output_of_dirichlet_data = func.apply(dirichlet_data)
                if np.all(float_cmp(output_of_dirichlet_data.to_numpy(), 0.)):
                    shifted_outputs += [func,]
                else:
                    shifted_outputs += [func + ConstantOperator(value=output_of_dirichlet_data, source=func.source),]
            outputs = shifted_outputs

        if len(outputs) == 0:
            output_functional = None
        elif len(outputs) == 1:
            output_functional = outputs[0]
        else:
            from pymor.operators.block import BlockColumnOperator
            output_functional = BlockColumnOperator(outputs)

        # visualizer
        if d == 1:
            # unshifted_visualizer = DuneGDT1dMatplotlibVisualizer(space) # only for stationary problems!
            unshifted_visualizer = DuneGDT1dasNumpyVisualizer(space, grid)
        else:
            unshifted_visualizer = DuneGDTK3dVisualizer(space) if is_jupyter() else DuneGDTParaviewVisualizer(space)

        if trivial_dirichlet_data:
            visualizer = unshifted_visualizer
        else:
            class ShiftedVisualizer(ImmutableObject):
                def __init__(self, visualizer, shift):
                    self.__auto_init(locals())

                def visualize(self, U, m, **kwargs):
                    return self.visualizer.visualize(U + self.shift, m, **kwargs)

            visualizer = ShiftedVisualizer(unshifted_visualizer, dirichlet_data)

        m  = StationaryModel(L, F, output_functional=output_functional, products=products, visualizer=visualizer,
                             name=f'{p.name}_CG')

        data = {'grid': grid,
                'boundary_info': boundary_info,
                'space': space,
                'interpolate': interpolate_single}

        if not trivial_dirichlet_data:
            data.update({
                'dirichlet_shift': dirichlet_data,
                'unshifted_visualizer': unshifted_visualizer,
                })

        return m, data


def discretize_instationary_cg(analytical_problem, diameter=None, domain_discretizer=None, grid_type=None,
                               grid=None, boundary_info=None, num_values=None, time_stepper=None, nt=None,
                               order=1, data_approximation_order=2, la_backend=Istl(),
                               advection_in_divergence_form=False, mu_energy_product=None):
    """Discretizes an |InstationaryProblem| with a |StationaryProblem| as stationary part
    using finite elements.

    Parameters
    ----------
    analytical_problem
        The |InstationaryProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If `None`, |discretize_domain_default| is used.
    grid_type
        If not `None`, this parameter is forwarded to `domain_discretizer` to specify
        the type of the generated |Grid|.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
        to be used by :class:`~pymor.models.basic.InstationaryModel.solve`.
    nt
        If `time_stepper` is not specified, the number of time steps for implicit
        Euler time stepping.
    preassemble
        If `True`, preassemble all operators in the resulting |Model|.

    Returns
    -------
    m
        The |Model| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
            :unassembled_m:  In case `preassemble` is `True`, the generated |Model|
                             before preassembling operators.
    """

    assert isinstance(analytical_problem, InstationaryProblem)
    assert isinstance(analytical_problem.stationary_part, StationaryProblem)
    assert (time_stepper is None) != (nt is None)

    p = analytical_problem

    assert not p.initial_data.parametric

    m, data = discretize_stationary_cg(p.stationary_part, diameter=diameter, domain_discretizer=domain_discretizer,
                                       grid_type=grid_type, grid=grid, boundary_info=boundary_info,
                                       order=order, data_approximation_order=data_approximation_order,
                                       la_backend=la_backend,
                                       advection_in_divergence_form=advection_in_divergence_form,
                                       mu_energy_product=mu_energy_product)

    # interpolate initial data
    df = DiscreteFunction(data['space'], la_backend)
    np_view = np.array(df.dofs.vector, copy=False)
    np_view[:] = p.initial_data.evaluate(data['space'].interpolation_points())[:].ravel()
    I = m.solution_space.make_array([DuneXTVector(df.dofs.vector),])

    if time_stepper is None:
        if p.stationary_part.diffusion is None:
            time_stepper = ExplicitEulerTimeStepper(nt=nt)
        else:
            time_stepper = ImplicitEulerTimeStepper(nt=nt)

    mass = m.l2_0_product

    m = InstationaryModel(operator=m.operator, rhs=m.rhs, mass=mass, initial_data=I, T=p.T,
                          products=m.products,
                          output_functional=m.output_functional,
                          time_stepper=time_stepper,
                          visualizer=m.visualizer,
                          num_values=num_values, name=f'{p.name}_CG')

    return m, data
