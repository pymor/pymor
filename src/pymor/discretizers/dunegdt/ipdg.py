from pymor.core.config import config, is_jupyter


if config.HAVE_DUNEGDT:
    import numpy as np
    from functools import partial
    from numbers import Number

    from dune.xt.grid import (
            ApplyOnCustomBoundaryIntersections,
            ApplyOnInnerIntersections,
            ApplyOnInnerIntersectionsOnce,
            Dim,
            DirichletBoundary,
            NeumannBoundary,
            RobinBoundary,
            Walker,
            )
    from dune.xt.functions import GridFunction as GF
    from dune.xt.la import Istl
    from dune.gdt import (
            DiscontinuousLagrangeSpace,
            DiscreteFunction,
            LocalCouplingIntersectionIntegralBilinearForm,
            LocalCouplingIntersectionRestrictedIntegralBilinearForm,
            LocalElementIntegralBilinearForm,
            LocalElementIntegralFunctional,
            LocalElementProductIntegrand,
            LocalIPDGBoundaryPenaltyIntegrand,
            LocalIPDGInnerPenaltyIntegrand,
            LocalIntersectionIntegralBilinearForm,
            LocalIntersectionIntegralFunctional,
            LocalIntersectionRestrictedIntegralBilinearForm,
            LocalIntersectionProductIntegrand,
            LocalLaplaceIPDGDirichletCouplingIntegrand,
            LocalLaplaceIPDGInnerCouplingIntegrand,
            LocalLaplaceIntegrand,
            LocalLinearAdvectionUpwindDirichletCouplingIntegrand,
            LocalLinearAdvectionUpwindInnerCouplingIntegrand,
            LocalLinearAdvectionIntegrand,
            LocalIntersectionRestrictedIntegralFunctional,
            MatrixOperator,
            VectorFunctional,
            estimate_combined_inverse_trace_inequality_constant,
            estimate_element_to_intersection_equivalence_constant,
            make_element_and_intersection_sparsity_pattern,
            )

    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import Function, ConstantFunction, LincombFunction
    from pymor.bindings.dunegdt import (
            DuneGDT1dMatplotlibVisualizer,
            DuneGDTK3dVisualizer,
            DuneGDTParaviewVisualizer,
            DuneXTMatrixOperator,
            DuneXTVector,
            DuneXTVectorSpace,
            )
    from pymor.discretizers.dunegdt.domaindiscretizers.default import discretize_domain_default
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import LincombOperator, VectorArrayOperator


    def discretize_stationary_ipdg(analytical_problem, diameter=None, domain_discretizer=None,
                                   grid_type=None, grid=None, boundary_info=None,
                                   order=1, data_approximation_order=2, la_backend=Istl(), symmetry_factor=1,
                                   weight_parameter=None, penalty_parameter=None):
        """Discretizes a |StationaryProblem| with dune-gdt using an interior penalty (IP) discontinuous Galerkin (DG)
           method based on Lagrange finite elements.

        The type of IPDG scheme is determined by `symmetry_factor` and `weight_parameter`:

        * with `weight_parameter==None` we obtain

          - `symmetry_factor==-1`: non-symmetric interior penalty scheme (NIPDG)
          - `symmetry_factor==0`: incomplete interior penalty scheme (IIPDG)
          - `symmetry_factor==1`: symmetric interior penalty scheme (SIPDG)

        * with `weight_parameter!=None`, we expect `weight_parameter` to be a |Parameter| compatible to the diffusion of
          the analytical problem, to create a nonparametric weight function (see below), and obtain

          - `symmetry_factor==1`: symmetric weighted interior penalty scheme (SWIPDG)

        Note that we currently only support linear advection, which is discretized with an upwind numerical flux.

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
        symmetry_factor
            Usually one of -1, 0, 1, determines the IPDG scheme (see above).
        weight_parameter
            Determines the IPDG scheme (see above), either None or compatible with the diffusion in the sense that:
            ```
            p = analytical_problem
            mu_weight = p.diffusion.parameters.parse(weight_parameter)
            weight = LincombFunction(p.diffusion.functions, p.diffusion.evaluate_coefficients(mu_weight))
        penalty_parameter
            Positive number to ensure coercivity of the resulting diffusion bilinear form. Is determined automatically
            if `None`.
            ```

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
        assert symmetry_factor in (-1, 0, 1)

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
                np_view[:] = func.evaluate(interpolation_points['scalar'])[:].ravel()
                return df
            elif func.shape_range == (d,):
                if not 'vector' in interpolation_space:
                    interpolation_space['vector'] = DiscontinuousLagrangeSpace(
                            grid, order=data_approximation_order, dim_range=Dim(d))
                if not 'vector' in interpolation_points:
                    interpolation_points['vector'] = interpolation_space['vector'].interpolation_points()
                df = DiscreteFunction(interpolation_space['vector'], la_backend)
                np_view = np.array(df.dofs.vector, copy=False)
                np_view[:] = func.evaluate(interpolation_points['vector'])[:].ravel()
                return df
            else:
                raise NotImplementedError(f'I do not know how to interpolate a function with {func.shape_range}!')

        def to_lincomb(func):
            if isinstance(func, LincombFunction):
                return func.functions, func.coefficients
            elif not isinstance(func, Function):
                func = ConstantFunction(value_array=func, dim_domain=d)
            return [func], [1.]

        def interpolate(func):
            functions, coefficients = to_lincomb(func)
            return [interpolate_single(ff) for ff in functions], coefficients

        # preparations for the actual discretization
        if order == data_approximation_order:
            if not 'scalar' in interpolation_space:
                interpolation_space['scalar'] = DiscontinuousLagrangeSpace(
                        grid, order=data_approximation_order, dim_range=Dim(1))
            space = interpolation_space['scalar']
        else:
            space = DiscontinuousLagrangeSpace(grid, order=order, dim_range=Dim(1))
        sparsity_pattern = make_element_and_intersection_sparsity_pattern(space)
        lhs_ops = []
        lhs_coeffs = []
        rhs_ops = []
        rhs_coeffs = []
        name = 'IIPDG'

        # diffusion part
        if p.diffusion:
            # penalty parameter for the diffusion part of the IPDG scheme
            if penalty_parameter is None:
                if symmetry_factor == -1:
                    name = 'NIPDG'
                    penalty_parameter = 1 # any positive number will do (the smaller the better)
                else:
                    name = 'SIPDG'
                    # TODO: check if we need to include diffusion for the coercivity here!
                    # TODO: each is a grid walk, compute this in one grid walk with the sparsity pattern
                    C_G = estimate_element_to_intersection_equivalence_constant(grid)
                    C_M_times_1_plus_C_T = estimate_combined_inverse_trace_inequality_constant(space)
                    penalty_parameter = C_G*C_M_times_1_plus_C_T
                    if symmetry_factor == 1:
                        penalty_parameter *= 4
            assert isinstance(penalty_parameter, Number)
            assert penalty_parameter > 0

            # weight for the diffusion part of the IPDG scheme (see above)
            if weight_parameter is None:
                weight = ConstantFunction(1, dim_domain=d)
            else:
                assert symmetry_factor == 1
                name = 'SWIPDG'
                mu_weight = p.diffusion.parameters.parse(weight_parameter)
                weight = LincombFunction(p.diffusion.functions, p.diffusion.evaluate_coefficients(mu_weight))
            weight = GF(grid, interpolate_single(weight), (Dim(d), Dim(d)))

            # contributions to the left hand side
            def make_diffusion_operator_parametric_part(func):
                op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                op += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GF(grid, func, (Dim(d), Dim(d)))))
                op += (LocalCouplingIntersectionIntegralBilinearForm(LocalLaplaceIPDGInnerCouplingIntegrand(
                            symmetry_factor, GF(grid, func, (Dim(d), Dim(d))), weight)),
                       {}, ApplyOnInnerIntersectionsOnce(grid))
                op += (LocalIntersectionIntegralBilinearForm(LocalLaplaceIPDGDirichletCouplingIntegrand(
                            symmetry_factor, GF(grid, func, (Dim(d), Dim(d))))),
                       {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                return op

            diffusion_funcs, diffusion_coeffs = interpolate(p.diffusion)
            lhs_ops += [make_diffusion_operator_parametric_part(func) for func in diffusion_funcs]
            lhs_coeffs += list(diffusion_coeffs)

            def make_diffusion_operator_nonparametric_part():
                op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                op += (LocalCouplingIntersectionIntegralBilinearForm(LocalIPDGInnerPenaltyIntegrand(
                            penalty_parameter, weight)),
                       {}, ApplyOnInnerIntersectionsOnce(grid))
                op += (LocalIntersectionIntegralBilinearForm(LocalIPDGBoundaryPenaltyIntegrand(
                            symmetry_factor, weight)),
                       {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                return op

            lhs_ops += [make_diffusion_operator_nonparametric_part()]
            lhs_coeffs += [1.]

            # contributions to the right hand side
            if p.dirichlet_data:
                def make_ipdg_dirichlet_penalty_functional(func):
                    op = VectorFunctional(grid, space, la_backend)
                    op += (LocalIntersectionIntegralFunctional(
                                LocalIntersectionProductIntegrand(GF(grid, penalty_parameter)).with_ansatz(GF(grid,
                                    func))), {},
                           ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                    return op

                dirichlet_funcs, dirichlet_coeffs = interpolate(p.dirichlet_data)
                rhs_ops += [make_ipdg_dirichlet_penalty_functional(func) for func in dirichlet_funcs]
                rhs_coeffs += list(dirichlet_coeffs)

                def make_laplace_ipdg_dirichlet_coupling_functional(dirichlet_func, diffusion_func):
                    op = VectorFunctional(grid, space, la_backend)
                    op += (LocalIntersectionIntegralFunctional(LocalLaplaceIPDGDirichletCouplingIntegrand(
                                symmetry_factor, GF(grid, diffusion_func, (Dim(d), Dim(d))), GF(grid, dirichlet_func))),
                           {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                    return op

                rhs_ops += [make_laplace_ipdg_dirichlet_coupling_functional(dirichlet_func, diffusion_func)
                            for diffusion_func in diffusion_funcs
                            for dirichlet_func in dirichlet_funcs]
                rhs_coeffs += [dirichlet_coeff*diffusion_coeff
                               for diffusion_coeff in diffusion_coeffs
                               for dirichlet_coeff in dirichlet_coeffs]

        # advection part
        if p.advection:
            # TODO: the filter is probably not runtime efficient, due to the temporary FieldVector/list conversion
            # x_local is in reference intersection coordinates
            def restrict_to_inflow(func):
                return lambda intersection, x_local: \
                        func.evaluate(intersection.to_global(x_local)).dot(intersection.unit_outer_normal(x_local)) < 0

            # we do not simply want to use interpolate() since we require the pyMOR function for the filter above
            # alongside the dune function

            # contributions to the left hand side
            def make_advection_operator(pymor_func, dune_func):
                op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                op += LocalElementIntegralBilinearForm(LocalLinearAdvectionIntegrand(GF(grid, dune_func)))
                    # logging_prefix='volume'))
                op += (LocalCouplingIntersectionRestrictedIntegralBilinearForm(restrict_to_inflow(pymor_func),
                    LocalLinearAdvectionUpwindInnerCouplingIntegrand(GF(grid, dune_func))), #logging_prefix='inner')),
                       {}, ApplyOnInnerIntersections(grid))
                op += (LocalIntersectionRestrictedIntegralBilinearForm(restrict_to_inflow(pymor_func),
                    LocalLinearAdvectionUpwindDirichletCouplingIntegrand(GF(grid, dune_func))),
                        # logging_prefix='dirichlet_lhs')),
                       {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                return op

            advection_funcs_P, advection_coeffs = to_lincomb(p.advection)
            advection_funcs_D = [interpolate_single(ff) for ff in advection_funcs_P]
            lhs_ops += [make_advection_operator(pymor_func, dune_func)
                        for pymor_func, dune_func in zip(advection_funcs_P, advection_funcs_D)]
            lhs_coeffs += list(advection_coeffs)

            # contributions to the right hand side
            if p.dirichlet_data:
                def make_advection_dirichlet_boundary_functional(
                        pymor_direction_func, dune_direction_func, dirichlet_func):
                    op = VectorFunctional(grid, space, la_backend)
                    op += (LocalIntersectionRestrictedIntegralFunctional(restrict_to_inflow(pymor_direction_func),
                            LocalLinearAdvectionUpwindDirichletCouplingIntegrand(
                                GF(grid, dune_direction_func), GF(grid, dirichlet_func))), #logging_prefix='dirichlet_rhs')),
                           {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                    return op

                if not p.diffusion:
                    dirichlet_funcs, dirichlet_coeffs = interpolate(p.dirichlet_data)
                rhs_ops += [make_advection_dirichlet_boundary_functional(
                    pymor_advection_func, dune_advection_func, dirichlet_func)
                            for dirichlet_func in dirichlet_funcs
                            for pymor_advection_func, dune_advection_func in zip(advection_funcs_P, advection_funcs_D)]
                rhs_coeffs += [advection_coeff*dirichlet_coeff
                               for dirichlet_coeff in dirichlet_coeffs
                               for advection_coeff in advection_coeffs]

        # reaction part
        def make_weighted_l2_operator(func):
             op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
             op += LocalElementIntegralBilinearForm(LocalElementProductIntegrand(GF(grid, func)))
             return op

        if p.reaction:
            reaction_funcs, reaction_coeffs = interpolate(p.reaction)
            lhs_ops += [make_weighted_l2_operator(func) for func in reaction_funcs]
            lhs_coeffs += list(reaction_coeffs)

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

            lhs_ops += [make_weighted_l2_robin_boundary_operator(func) for func in robin_parameter_funcs]
            lhs_coeffs += list(robin_parameter_coeffs)

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

        # products
        l2_product = make_weighted_l2_operator(1)
        h1_product = make_weighted_l2_operator(1)
        h1_product += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GF(grid, 1, (Dim(d), Dim(d)))))
        if p.diffusion:
            weighted_h1_semi_penalty_product = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
            weighted_h1_semi_penalty_product += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(weight))
            weighted_h1_semi_penalty_product += (
                    LocalCouplingIntersectionIntegralBilinearForm(LocalIPDGInnerPenaltyIntegrand(
                        penalty_parameter, weight)),
                    {}, ApplyOnInnerIntersectionsOnce(grid))
            weighted_h1_semi_penalty_product += (
                    LocalIntersectionIntegralBilinearForm(LocalIPDGBoundaryPenaltyIntegrand(
                        symmetry_factor, weight)),
                    {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))

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
        for op in lhs_ops:
            walker.append(op)
        for op in rhs_ops:
            walker.append(op)
        walker.append(l2_product)
        walker.append(h1_product)
        if p.diffusion:
            walker.append(weighted_h1_semi_penalty_product)
        for op in outputs:
            walker.append(op)
        walker.walk(thread_parallel=False) # support not stable/enabled yet

        # wrap everything as pyMOR operators
        lhs_ops = [DuneXTMatrixOperator(op.matrix) for op in lhs_ops]
        L = LincombOperator(operators=lhs_ops, coefficients=lhs_coeffs, name='ellipticOperator')

        rhs_ops = [VectorArrayOperator(lhs_ops[0].range.make_array([DuneXTVector(op.vector)])) for op in rhs_ops]
        F = LincombOperator(operators=rhs_ops, coefficients=rhs_coeffs, name='rhsOperator')

        products = {'l2': DuneXTMatrixOperator(l2_product.matrix),
                    'h1': DuneXTMatrixOperator(h1_product.matrix)}
        if p.diffusion:
            products['weighted_h1_semi_penalty'] = DuneXTMatrixOperator(weighted_h1_semi_penalty_product.matrix)

        outputs = [DuneXTVector(op) for op in outputs]
        if len(outputs) == 0:
            output_functional = None
        elif len(outputs) == 1:
            output_functional = outputs[0]
        else:
            from pymor.operators.block import BlockColumnOperator
            output_functional = BlockColumnOperator(outputs)

        # visualizer
        if d == 1:
            visualizer = DuneGDT1dMatplotlibVisualizer(space)
        else:
            visualizer = DuneGDTK3dVisualizer(space) if is_jupyter() else DuneGDTParaviewVisualizer(space)

        m  = StationaryModel(L, F, output_functional=output_functional, products=products, visualizer=visualizer,
                             name=f'{p.name}_{name}')

        data = {'grid': grid, 'boundary_info': boundary_info}

        return m, data


