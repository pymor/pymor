import numpy as np

from pymor.algorithms.preassemble import preassemble as preassemble_
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.bindings.dunegdt import DuneXTMatrixOperator
from pymor.discretizers.dunegdt.problems import StationaryDuneProblem
from pymor.discretizers.dunegdt.cg import _discretize_stationary_cg_dune
from pymor.discretizers.dunegdt.ipdg import (
    _discretize_stationary_ipdg_dune, _IP_estimate_penalty_parameter, _IP_scheme_id)
from pymor.models.basic import StationaryModel
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import VectorOperator, LincombOperator

from dune.xt.grid import (
    AllNeumannBoundaryInfo,
    ApplyOnCustomBoundaryIntersections,
    CouplingIntersection,
    Dim,
    DirichletBoundary,
    Cube,
    Simplex,
    Walker,
    make_cube_grid,
    make_cube_dd_grid,
)
from dune.xt.functions import GridFunction as GF
from dune.xt.la import Istl
from dune.gdt import (
    BilinearForm,
    LocalIntersectionIntegralBilinearForm,
    LocalCouplingIntersectionIntegralBilinearForm,
    LocalLaplaceIPDGInnerCouplingIntegrand,
    LocalLaplaceIPDGDirichletCouplingIntegrand,
    LocalIPDGInnerPenaltyIntegrand,
    LocalIPDGBoundaryPenaltyIntegrand,
    MatrixOperator,
    make_coupling_sparsity_pattern,
)


def discretize_stationary_ipld3g(
    analytical_problem,
    macro_diameter=None,
    num_local_refinements=None,
    order=1,
    data_approximation_order=2,
    la_backend=Istl(),
    symmetry_factor=1,
    weight_parameter=None,
    penalty_parameter=None,
    preassemble=True,
    locally_continuous=True):
    '''
    interior penalty localized domain decomposition discontinuous Galerkin = IPLDDDG = IPLD3G
    '''

    assert isinstance(analytical_problem, StationaryProblem)
    assert analytical_problem.advection is None, "Not implemented yet!"
    assert analytical_problem.diffusion is not None, "Not implemented yet!"
    assert analytical_problem.robin_data is None, "Not implemented yet!"
    assert analytical_problem.neumann_data is None, "Not implemented yet!"
    assert analytical_problem.dirichlet_data is None, "Not implemented yet!"
    assert analytical_problem.outputs is None, "Not implemented yet!"

    # we only use the macro problem for the macro grid, the boundary info and sanity checks
    if macro_diameter is None:
        macro_diameter = analytical_problem.domain.dim/(3*np.sqrt(2))
    macro_dune_problem = StationaryDuneProblem.from_pymor(
        analytical_problem, 0, diameter=macro_diameter, grid_type='cube')
    # TODO: enable 'simplex' as macro and local grid_type
    macro_grid, macro_boundary_info = macro_dune_problem.grid, macro_dune_problem.boundary_info
    del macro_dune_problem
    d = macro_grid.dimension

    if num_local_refinements is None:
        num_local_refinements = 5
    dd_grid = make_cube_dd_grid(macro_grid, local_element_type=Cube(), num_refinements=num_local_refinements)

    # we obtain the discrete model by
    # - collecting the coupling contributions in a separate BlockOperator
    M = dd_grid.num_subdomains
    local_ops = np.empty((M, M), dtype=object)
    local_rhs = np.empty((M, 1), dtype=object)
    coupling_ops = np.empty((M, M), dtype=object)

    # - building local models from discretizations of localized problems
    #   without essential boundary conditions
    #   TODO: adapt for non-trivial Neumann and/or Robin boundary data
    local_analytical_problem = analytical_problem.with_(dirichlet_data=None)
    #   * convert/interpolate data functions on local grids
    local_problems = [StationaryDuneProblem.from_pymor(
        local_analytical_problem, data_approximation_order,
        grid=dd_grid.local_grid(I), boundary_info=AllNeumannBoundaryInfo(dd_grid.local_grid(I)))
                      for I in range(M)]
    #   * discretize locally
    local_models = []
    local_models_data = []
    for I in range(M):
        if locally_continuous:
            local_model, local_model_data = _discretize_stationary_cg_dune(
                local_problems[I], order=order, la_backend=la_backend, preassemble=preassemble)
        else:
            local_model, local_model_data = _discretize_stationary_ipdg_dune(
                local_problems[I], order=order, la_backend=la_backend, symmetry_factor=symmetry_factor,
                weight_parameter=weight_parameter, penalty_parameter=penalty_parameter,
                preassemble=preassemble)
        local_models.append(local_model)
        local_models_data.append(local_model_data)
        local_ops[I][I] = local_model.operator.with_(name=f'volume_part_{I}')
        local_rhs[I] = local_model.rhs

    #   (from here on, we basically follow discretize_stationary_ipdg)
    IP_scheme_ID = _IP_scheme_id(symmetry_factor, weight_parameter)  # performs some checks
    # penalty parameter for the diffusion part of the IPDG scheme
    if penalty_parameter is None:
        # TODO: add missing min diffusion estimate, see discretize_stationary_ipdg
        if locally_continuous:
            for I in range(M):
                local_models_data[I]['IP_penalty_parameter'] = _IP_estimate_penalty_parameter(
                    local_problems[I].grid, local_models_data[I]['space'],
                    symmetry_factor, weight_parameter)
        penalty_parameter = np.max([data['IP_penalty_parameter'] for data in local_models_data])
    # weight for the diffusion part of the IPDG scheme (see above)
    if weight_parameter is None:
        local_weights = [GF(p.grid, 1, (Dim(d), Dim(d))) for p in local_problems]
    else:
        mu_weight = local_problems[0].diffusion.parameters.parse(weight_parameter)
        local_weights = [p.diffusion.assemble(mu_weight) for p in local_problems]

    # - weak enforcing of Dirichlet boundary values
    for I in dd_grid.boundary_subdomains:
        local_grid = local_problems[I].grid
        walker = Walker(local_grid)
        boundary_info = dd_grid.macro_based_boundary_info(I, macro_boundary_info)

        def make_boundary_contributions_parametric_part(func):
            bf = BilinearForm(local_grid)
            bf += (LocalIntersectionIntegralBilinearForm(
                LocalLaplaceIPDGDirichletCouplingIntegrand(
                    symmetry_factor,
                    GF(local_grid, func, (Dim(d), Dim(d))))),
                   ApplyOnCustomBoundaryIntersections(
                       local_grid,
                       boundary_info,
                       DirichletBoundary()))
            op = MatrixOperator(local_grid, local_models_data[I]['space'], local_models_data[I]['space'], la_backend,
                                local_models_data[I]['sparsity_pattern'])
            op.append(bf)
            walker.append(op)
            return op

        def make_boundary_contributions_nonparametric_part():
            bf = BilinearForm(local_grid)
            bf += (LocalIntersectionIntegralBilinearForm(
                LocalIPDGBoundaryPenaltyIntegrand(
                    penalty_parameter,
                    local_weights[I])),
                   ApplyOnCustomBoundaryIntersections(
                       local_grid,
                       boundary_info,
                       DirichletBoundary()))
            op = MatrixOperator(local_grid, local_models_data[I]['space'], local_models_data[I]['space'], la_backend,
                                local_models_data[I]['sparsity_pattern'])
            op.append(bf)
            walker.append(op)
            return op

        ops = [make_boundary_contributions_parametric_part(func)
               for func in local_problems[I].diffusion.functions] + [make_boundary_contributions_nonparametric_part(),]
        coeffs = list(local_problems[I].diffusion.coefficients) + [1,]

        walker.walk(False)  # not supported yet

        local_ops[I][I] += LincombOperator(
            operators=[DuneXTMatrixOperator(op.matrix, name=f'boundary_part_{I}') for op in ops],
            coefficients=coeffs)

    # - coupling of the local models by IP techniques
    for I in range(M):
        for J in dd_grid.neighbors(I):
            if I < J:  # treat each coupling only once, but from both sides
                coupling_grid = dd_grid.coupling_grid(I, J)
                walker = Walker(coupling_grid)
                coupling_sparsity_pattern = make_coupling_sparsity_pattern(
                    local_models_data[I]['space'],
                    local_models_data[J]['space'],
                    coupling_grid)

                def make_coupling_ops_from_bilinear_form(bf):
                    op_I_I = MatrixOperator(coupling_grid, local_models_data[I]['space'], local_models_data[I]['space'], local_models_data[I]['sparsity_pattern'])
                    op_I_J = MatrixOperator(coupling_grid, local_models_data[I]['space'], local_models_data[J]['space'], coupling_sparsity_pattern)
                    op_J_I = MatrixOperator(coupling_grid, local_models_data[J]['space'], local_models_data[I]['space'], coupling_sparsity_pattern)  # TODO: transpose pattern?!
                    op_J_J = MatrixOperator(coupling_grid, local_models_data[J]['space'], local_models_data[J]['space'], local_models_data[J]['sparsity_pattern'])
                    op_I_I.append(bf, {}, (False, True , False, False, False, False))  # volume, in_in, in_out, out_in, out_out, boundary
                    op_I_J.append(bf, {}, (False, False, True , False, False, False))
                    op_J_I.append(bf, {}, (False, False, False, True , False, False))
                    op_J_J.append(bf, {}, (False, False, False, False, True , False))
                    walker.append(op_I_I)
                    walker.append(op_I_J)
                    walker.append(op_J_I)
                    walker.append(op_J_J)
                    return op_I_I, op_I_J, op_J_I, op_J_J

                def make_coupling_contributions_parametric_part(func_in, func_out):
                    bf = BilinearForm(coupling_grid)
                    bf += LocalCouplingIntersectionIntegralBilinearForm(
                        LocalLaplaceIPDGInnerCouplingIntegrand(
                            symmetry_factor,
                            GF(local_problems[I].grid, func_in, (Dim(d), Dim(d))),
                            GF(local_problems[I].grid, func_out, (Dim(d), Dim(d))),
                            local_weights[I],
                            local_weights[J],
                            intersection_type=CouplingIntersection(dd_grid)))
                    return make_coupling_ops_from_bilinear_form(bf)

                def make_coupling_contributions_nonparametric_part():
                    bf = BilinearForm(coupling_grid)
                    bf += (LocalCouplingIntersectionIntegralBilinearForm(
                        LocalIPDGInnerPenaltyIntegrand(
                            penalty_parameter,
                            local_weights[I],
                            local_weights[J],
                            intersection_type=CouplingIntersection(dd_grid))))
                    return make_coupling_ops_from_bilinear_form(bf)

                ops_I_I = []
                ops_I_J = []
                ops_J_I = []
                ops_J_J = []
                coeffs = []
                for diff_in, coeff_in, diff_out, coeff_out in zip(
                    local_problems[I].diffusion.functions,
                    local_problems[I].diffusion.coefficients,
                    local_problems[J].diffusion.functions,
                    local_problems[J].diffusion.coefficients):
                    assert coeff_in == coeff_out
                    op_I_I, op_I_J, op_J_I, op_J_J = make_coupling_contributions_parametric_part(
                        diff_in, diff_out)
                    ops_I_I.append(op_I_I)
                    ops_I_J.append(op_I_J)
                    ops_J_I.append(op_J_I)
                    ops_J_J.append(op_J_J)
                    coeffs.append(coeff_in)
                op_I_I, op_I_J, op_J_I, op_J_J = make_coupling_contributions_nonparametric_part()
                ops_I_I.append(op_I_I)
                ops_I_J.append(op_I_J)
                ops_J_I.append(op_J_I)
                ops_J_J.append(op_J_J)
                coeffs.append(1.)

                walker.walk(False)  # not yet supported

                for (i, j, ops) in ((I, I, ops_I_I), (I, J, ops_I_J), (J, I, ops_J_I), (J, J, ops_J_J)):
                    if coupling_ops[i][j] is None:
                        coupling_ops[i][j] = LincombOperator(
                            operators=[DuneXTMatrixOperator(op.matrix, name=f'coupling_part_from_{I}_{J}_{i}_{j}') for op in ops],
                            coefficients=list(coeffs))
                    else:
                        coupling_ops[i][j] = coupling_ops[i][j].with_(
                            operators=list(coupling_ops[i][j].operators) +
                            [DuneXTMatrixOperator(op.matrix, name=f'coupling_part_from_{I}_{J}_{i}_{j}') for op in ops],
                            coefficients=list(coupling_ops[i][j].coefficients) + list(coeffs)
                        )

    lhs_op = BlockOperator(local_ops) + BlockOperator(coupling_ops)
    rhs_op = BlockOperator(local_rhs)
    m = StationaryModel(lhs_op, rhs_op, name=f'{analytical_problem.name}_P{order}{IP_scheme_ID[:-2]}LD3G')

    data = {'dd_grid': dd_grid, 'macro_boundary_info': macro_boundary_info}

    if preassemble:
        data['unassembled_m'] = m
        m = preassemble_(m)

    return m, data
