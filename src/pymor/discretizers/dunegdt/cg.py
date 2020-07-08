from pymor.core.config import config, is_jupyter


if config.HAVE_DUNEGDT:
    import numpy as np
    from functools import partial

    from dune.xt.grid import (
            ApplyOnAllIntersectionsOnce,
            ApplyOnCustomBoundaryIntersections,
            Dim,
            NeumannBoundary,
            RobinBoundary,
            Walker,
            )
    from dune.xt.functions import GridFunction as GF
    from dune.xt.la import Istl
    from dune.gdt import (
            ContinuousLagrangeSpace,
            DirichletConstraints,
            DiscontinuousLagrangeSpace,
            DiscreteFunction,
            LocalElementIntegralBilinearForm,
            LocalElementIntegralFunctional,
            LocalElementProductIntegrand,
            LocalIntersectionProductIntegrand,
            LocalLaplaceIntegrand,
            LocalLinearAdvectionIntegrand,
            MatrixOperator,
            VectorFunctional,
            make_element_sparsity_pattern,
            )

    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import Function, ConstantFunction, LincombFunction
    from pymor.bindings.dunegdt import (
            DuneGDTK3dVisualizer,
            DuneGDTParaviewVisualizer,
            DuneXTMatrixOperator,
            DuneXTVector,
            DuneXTVectorSpace,
            )
    from pymor.discretizers.dunegdt.domaindiscretizers.default import discretize_domain_default
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import LincombOperator, VectorArrayOperator


    def discretize_stationary_cg(analytical_problem, diameter=None, domain_discretizer=None,
                                 grid_type=None, grid=None, boundary_info=None,
                                 order=1, data_approximation_order=2, la_backend=Istl()):
        """Discretizes a |StationaryProblem| with dune-gdt using continuous Lagrange finite elements.

        Note: all data functions are replaced by their respective non-conforming interpolations. This allows to simply
              use pyMORs data |Function|s at the expense of one DoF vector for each data function during discretization.

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

        Returns
        -------
        m
            The |Model| that has been generated.
        data
            Dictionary with the following entries:

                :grid:           The generated |Grid|.
                :boundary_info:  The generated |BoundaryInfo|.
        """

        assert isinstance(analytical_problem, StationaryProblem)
        assert grid is None or boundary_info is not None
        assert boundary_info is None or grid is not None
        assert grid is None or domain_discretizer is None
        assert grid_type is None or grid is None

        p = analytical_problem
        d = p.domain.dim

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
        interpolation_space = {}
        interpolation_points = {}

        def interpolate_single(func):
            assert isinstance(func, Function)
            if func.shape_range in ((), (1,)):
                if not 'scalar' in interpolation_space:
                    interpolation_space['scalar'] = DiscontinuousLagrangeSpace(
                            grid, order=data_approximation_order, dim_range=Dim(1))
                if not 'scalar' in interpolation_points:
                    interpolation_points['scalar'] = interpolation_space['scalar'].interpolation_points()
                df = DiscreteFunction(interpolation_space['scalar'], la_backend)
                np_view = np.array(df.dofs.vector, copy=False)
                np_view[:] = func.evaluate(interpolation_points['scalar'])[:]
                return df
            elif func.shape_range == (d,):
                if not 'vector' in interpolation_space:
                    interpolation_space['vector'] = DiscontinuousLagrangeSpace(
                            grid, order=data_approximation_order, dim_range=Dim(d))
                if not 'vector' in interpolation_points:
                    interpolation_points['vector'] = interpolation_space['vector'].interpolation_points()
                df = DiscreteFunction(interpolation_space['vector'], la_backend)
                np_view = np.array(df.dofs.vector, copy=False)
                np_view[:] = func.evaluate(interpolation_points['vector'])[:]
                return df
            else:
                raise NotImplementedError(f'I do not know how to interpolate a function with {func.shape_range}!')

        def interpolate(func):
            if isinstance(func, LincombFunction):
                return [interpolate_single(ff) for ff in func.functions], func.coefficients
            elif not isinstance(func, Function):
                func = ConstantFunction(value_array=func, dim_domain=d)
            return [interpolate_single(func)], [1]

        # preparations for the actual discretization
        space = ContinuousLagrangeSpace(grid, order=order, dim_range=Dim(1))
        sparsity_pattern = make_element_sparsity_pattern(space)
        constrained_lhs_ops = []
        constrained_lhs_coeffs = []
        unconstrained_lhs_ops = []
        unconstrained_lhs_coeffs = []
        rhs_ops = []
        rhs_coeffs = []

        # diffusion part
        def make_diffusion_operator(func):
            op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
            op += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GF(grid, func, (Dim(d), Dim(d)))))
            return op

        if p.diffusion:
            funcs, coeffs = interpolate(p.diffusion)
            constrained_lhs_ops += [make_diffusion_operator(func) for func in funcs]
            constrained_lhs_coeffs += list(coeffs)

        # advection part
        if p.advection:
            def make_advection_operator(func):
                 op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                 op += LocalElementIntegralBilinearForm(LocalLinearAdvectionIntegrand(GF(grid, func)))
                 return op

            funcs, coeffs = interpolate(p.advection)
            constrained_lhs_ops += [make_advection_operator(func) for func in funcs]
            constrained_lhs_coeffs += list(coeffs)

        # reaction part
        def make_weighted_l2_operator(func):
             op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
             op += LocalElementIntegralBilinearForm(LocalElementProductIntegrand(GF(grid, func)))
             return op

        if p.reaction:
            funcs, coeffs = interpolate(p.reaction)
            constrained_lhs_ops += [make_weighted_l2_operator(func) for func in funcs]
            constrained_lhs_coeffs += list(coeffs)

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

        # neumann boundaries
        if p.neumann_data:
            def make_l2_neumann_boundary_functional(func):
                op = VectorFunctional(grid, space, la_backend)
                op += (LocalIntersectionIntegralFunctional(
                            LocalIntersectionProductIntegrand(GF(grid, 1)).with_ansatz(-GF(grid, func))), {},
                       ApplyOnCustomBoundaryIntersections(grid, boundary_info, NeumannBoundary()))
                return op

            neumann_data_funcs, neumann_data_coeffs = interpolate(p.neumann_data)
            rhs_ops += [make_l2_neumann_boundary_functional(func) for func in neumann_data_funcs]
            rhs_coeffs += list(neumann_data_coeffs)

        # dirichlet boundaries will be handled further below ...

        # products
        l2_product = make_weighted_l2_operator(1)
        h1_semi_product = make_diffusion_operator(1)

        # output functionals
        outputs = []
        if p.outputs:
            if any(v[0] not in ('l2', 'l2_boundary') for v in p.outputs):
                raise NotImplementedError(f'I do not know how to discretize a {v[0]} output!')
            for output_type, output_data in p.outputs:
                output_data = interpolate_single(output_data)
                if output_type == 'l2':
                    op = VectorFunctional(grid, space, la_backend)
                    op += LocalElementIntegralFunctional(LocalElementProductIntegrand(grid).with_ansatz(output_data))
                    outputs.append(op)
                elif output_type == 'l2_boundary':
                    op = VectorFunctional(grid, space, la_backend)
                    op += (LocalIntersectionIntegralBilinearForm(
                            LocalIntersectionProductIntegrand(grid).with_ansatz(output_data)), {},
                            ApplyOnAllIntersectionsOnce(grid))
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
        for op in outputs:
            walker.append(op)
        walker.walk(thread_parallel=False) # support not stable/enabled yet

        # apply the dirichlet constraints
        for op in constrained_lhs_ops:
            dirichlet_constraints.apply(op.matrix, only_clear=True, ensure_symmetry=True)
        for op in rhs_ops:
            dirichlet_constraints.apply(op.vector) # sets to zero
        l2_0_product = MatrixOperator(grid, space, space, l2_product.matrix.copy()) # using operators here just for
        h1_0_semi_product = MatrixOperator(grid, space, space, h1_semi_product.matrix.copy()) # unified handling below
        dirichlet_constraints.apply(l2_0_product.matrix, ensure_symmetry=True)
        dirichlet_constraints.apply(h1_0_semi_product.matrix, ensure_symmetry=True)

        # ... and finally handle the dirichlet boundary values
        # - lhs contribution: a matrix to hold the unit rows/cols
        op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
        dirichlet_constraints.apply(op.matrix)
        lhs_ops = [op] + constrained_lhs_ops + unconstrained_lhs_ops
        lhs_coeffs = [1.] + constrained_lhs_coeffs + unconstrained_lhs_coeffs

        # - rhs contribution: a vector to hold the boundary values
        if p.dirichlet_data:
            dirichlet_data = interpolate_single(p.dirichlet_data)

            op = VectorFunctional(grid, space, la_backend) # just for the vector and unified handling below
            dirichlet_DoFs = dirichlet_constraints.dirichlet_DoFs
            op.vector[dirichlet_Dofs] = dirichlet_data.dofs.vector[dirichlet_DoFs]

            rhs_ops += [op]
            rhs_coeffs += [1.]

        # wrap everything as pyMOR operators
        lhs_ops = [DuneXTMatrixOperator(op.matrix) for op in lhs_ops]
        L = LincombOperator(operators=lhs_ops, coefficients=lhs_coeffs, name='ellipticOperator')

        rhs_ops = [VectorArrayOperator(lhs_ops[0].range.make_array([DuneXTVector(op.vector)])) for op in rhs_ops]
        F = LincombOperator(operators=rhs_ops, coefficients=rhs_coeffs, name='rhsOperator')

        products = {'h1': (DuneXTMatrixOperator(l2_product.matrix)
                           + DuneXTMatrixOperator(h1_semi_product.matrix)).assemble(),
                    'h1_semi': DuneXTMatrixOperator(h1_semi_product.matrix),
                    'l2': DuneXTMatrixOperator(l2_product.matrix),
                    'h1_0': (DuneXTMatrixOperator(l2_0_product.matrix)
                             + DuneXTMatrixOperator(h1_0_semi_product.matrix)).assemble(),
                    'h1_0_semi': DuneXTMatrixOperator(h1_0_semi_product.matrix),
                    'l2_0': DuneXTMatrixOperator(l2_0_product.matrix)}

        outputs = [DuneXTVector(op) for op in outputs]
        if len(outputs) == 0:
            output_functional = None
        elif len(outputs) == 1:
            output_functional = outputs[0]
        else:
            from pymor.operators.block import BlockColumnOperator
            output_functional = BlockColumnOperator(outputs)

        # visualizer
        visualizer = DuneGDTK3dVisualizer(space) if is_jupyter() else DuneGDTParaviewVisualizer(space)

        m  = StationaryModel(L, F, output_functional=output_functional, products=products, visualizer=visualizer,
                             name=f'{p.name}_CG')

        data = {'grid': grid, 'boundary_info': boundary_info}

        return m, data


