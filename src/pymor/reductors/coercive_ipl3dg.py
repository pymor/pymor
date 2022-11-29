import numpy as np
from copy import deepcopy

from pymor.core.base import ImmutableObject

from pymor.algorithms.projection import project
from pymor.models.basic import StationaryModel
from pymor.operators.block import BlockOperator, BlockColumnOperator
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.operators.constructions import VectorArrayOperator, IdentityOperator
from pymor.parameters.functionals import ParameterFunctional
from pymor.reductors.basic import extend_basis
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.reductors.residual import ResidualReductor, ResidualOperator


class CoerciveIPLD3GRBReductor(CoerciveRBReductor):
    def __init__(self, fom, dd_grid, local_bases=None):
        self.fom = fom
        self.solution_space = fom.solution_space
        self.S = self.solution_space.empty().num_blocks
        self._last_rom = None

        # TODO: assertions for local_bases
        self.local_bases = local_bases or [self.solution_space.empty().block(ss).empty()
                                           for ss in range(self.S)]

        # element patches for online enrichment
        self.element_patches = construct_element_patches(dd_grid)

        self.patch_models = []
        self.patch_mappings_to_global = []
        self.patch_mappings_to_local = []
        for element_patch in self.element_patches:
            patch_model, local_to_global, global_to_local = construct_local_model(
                element_patch, fom.operator, fom.rhs, dd_grid.neighbors)
            self.patch_models.append(patch_model)
            self.patch_mappings_to_global.append(local_to_global)
            self.patch_mappings_to_local.append(global_to_local)

        # node patches for estimation
        # NOTE: currently the estimator domains are node patches
        inner_node_patches = construct_inner_node_patches(dd_grid)
        estimator_domains = add_element_neighbors(dd_grid, inner_node_patches)

        self.estimator_data = {}
        self.local_residuals = []
        for associated_element, elements in estimator_domains.items():
            local_model, local_to_global, global_to_local = construct_local_model(
                elements, fom.operator, fom.rhs, dd_grid.neighbors,
                block_prod=fom.products['h1'], lincomb_outside=True)
            bases_in_local_domain = [self.local_bases[el] for el in elements]
            basis = local_model.solution_space.make_array(bases_in_local_domain)
            residual_reductor = ResidualReductor(basis,
                                                 local_model.operator,
                                                 local_model.rhs,
                                                 product=local_model.products['h1'],
                                                 riesz_representatives=True)
            node_elements = inner_node_patches[associated_element]
            # TODO: vectorize the following
            local_node_elements = [global_to_local(el) for el in node_elements]
            self.estimator_data[associated_element] = (elements, node_elements,
                                                       local_node_elements)
            self.local_residuals.append(residual_reductor)

    def add_partition_of_unity(self):
        for I in range(self.S):
            # TODO: find the correct partition of unity here
            pass

    def add_global_solutions(self, us):
        assert us in self.fom.solution_space
        for I in range(self.S):
            us_block = us.block(I)
            try:
                extend_basis(us_block, self.local_bases[I])
            except:
                print('Extension failed')

    def add_local_solutions(self, I, u):
        try:
            extend_basis(u, self.local_bases[I])
        except:
            print('Extension failed')

    def enrich_all_locally(self, mu, use_global_matrix=False):
        for I in range(self.S):
            _ = self.enrich_locally(I, mu, use_global_matrix=use_global_matrix)

    def enrich_locally(self, I, mu, use_global_matrix=False):
        if self._last_rom is None:
            _ = self.reduce()
        mu_parsed = self.fom.parameters.parse(mu)
        current_solution = self._last_rom.solve(mu)
        mapping_to_global = self.patch_mappings_to_global[I]
        mapping_to_local = self.patch_mappings_to_local[I]
        patch_model = self.patch_models[I]

        if use_global_matrix:
            # use global matrix
            print("Warning: you are using the global operator here")

            # this part is using the global discretization
            current_solution_h = self.reconstruct(current_solution)
            a_u_v_global = self.fom.operator.apply(current_solution_h, mu_parsed)

            a_u_v_restricted_to_patch = []
            for i in range(len(patch_model.operator.range.subspaces)):
                i_global = mapping_to_global(i)
                a_u_v_restricted_to_patch.append(a_u_v_global.block(i_global))
            a_u_v_restricted_to_patch = patch_model.operator.range.make_array(
                a_u_v_restricted_to_patch)
            a_u_v_as_operator = VectorOperator(a_u_v_restricted_to_patch)
        else:
            u_restricted_to_patch = []
            for i_loc in range(len(patch_model.operator.range.subspaces)):
                i_global = mapping_to_global(i_loc)
                basis = self.reduced_local_bases[i_global]
                u_restricted_to_patch.append(
                    basis.lincomb(current_solution.block(i_global).to_numpy()))
            current_solution_on_patch = patch_model.operator.range.make_array(
                u_restricted_to_patch)

            new_op = remove_irrelevant_coupling_from_patch_operator(patch_model,
                                                                    mapping_to_global)
            a_u_v = new_op.apply(current_solution_on_patch, mu_parsed)
            a_u_v_as_operator = VectorOperator(a_u_v)

        patch_model_with_correction = patch_model.with_(
            rhs=patch_model.rhs - a_u_v_as_operator)
        # TODO: think about removing boundary dofs for the rhs
        phi = patch_model_with_correction.solve(mu)
        self.add_local_solutions(I, phi.block(mapping_to_local(I)))
        return phi

    def basis_length(self):
        return [len(self.local_bases[I]) for I in range(self.S)]

    def reduce(self, dims=None):
        assert dims is None, 'we cannot yet reduce to subbases'

        if self._last_rom is None or sum(self.basis_length()) > self._last_rom_dims:
            self.reduced_residuals = self._reduce_residuals()
            self._last_rom = self._reduce()
            self._last_rom_dims = sum(self.basis_length())
            # self.reduced_local_basis is required to perform multiple local enrichments
            # TODO: check how to fix this better
            self.reduced_local_bases = deepcopy(self.local_bases)

        return self._last_rom

    def project_operators(self):
        # this is for BlockOperator(LincombOperators)
        assert isinstance(self.fom.operator, BlockOperator)

        # TODO: think about not projection the BlockOperator, but instead get rid
        # of the Block structure (like usual in localized MOR)
        # or use methodology of Stage 2 in TSRBLOD

        # see PR #894 in pymor
        projected_operator = project_block_operator(self.fom.operator, self.local_bases,
                                                    self.local_bases)
        projected_rhs = project_block_rhs(self.fom.rhs, self.local_bases)

        projected_products = {k: project_block_operator(v, self.local_bases,
                                                        self.local_bases)
                              for k, v in self.fom.products.items()}

        # TODO: project output functional

        projected_operators = {
            'operator':          projected_operator,
            'rhs':               projected_rhs,
            'products':          projected_products,
            'output_functional': None
        }
        return projected_operators

    def _reduce_residuals(self):
        # reduced_residuals = [local_residual.reduce() for local_residual in self.local_residuals]
        # NOTE: the above does not make sense because there is no system reduce involved.
        # we need to use SimpleCoerciveReductor which also uses project_block_operator and friends
        # instead: just use the FOM parts for now without reduction

        reduced_residuals = [local_residual for local_residual in self.local_residuals]
        return reduced_residuals

    def assemble_error_estimator(self):
        estimators = {}
        estimators['global'] = GlobalEllipticEstimator(self.fom)
        estimators['local'] = EllipticIPLRBEstimator(self.estimator_data,
                                                     self.reduced_residuals,
                                                     self.S)
        return estimators

    def reduce_to_subbasis(self, dims):
        raise NotImplementedError

    def reconstruct(self, u_rom):
        u_ = []
        for I in range(self.S):
            basis = self.reduced_local_bases[I]
            u_I = u_rom.block(I)
            u_.append(basis.lincomb(u_I.to_numpy()))
        return self.fom.solution_space.make_array(u_)

    def from_patch_to_global(self, I, u_patch):
        # a function that construct a globally defined u from a patch u
        # only here for visualization purposes. not relevant for reductor
        u_global = []
        for i in range(self.S):
            i_loc = self.patch_mappings_to_local[I](i)
            if i_loc >= 0:
                u_global.append(u_patch.block(i_loc))
            else:
                # the multiplication with 1e-4 is only there for visualization
                # purposes. Otherwise, the colorbar gets scaled up until 1.
                # which is bad
                # TODO: fix colorscale for the plot
                u_global.append(self.solution_space.subspaces[i].ones()*1e-4)
        return self.solution_space.make_array(u_global)


def construct_local_model(local_elements, block_op, block_rhs, neighbors, block_prod=None,
                          lincomb_outside=False):
    def local_to_global_mapping(i):
        return local_elements[i]

    def global_to_local_mapping(i):
        for i_, j in enumerate(local_elements):
            if j == i:
                return i_
        return -1

    S_patch = len(local_elements)
    patch_op = np.empty((S_patch, S_patch), dtype=object)
    patch_rhs = np.empty(S_patch, dtype=object)
    patch_prod = np.empty((S_patch, S_patch), dtype=object)
    blocks_op = block_op.blocks
    blocks_rhs = block_rhs.blocks
    blocks_prod = block_prod.blocks if block_prod else None
    for ii in range(S_patch):
        I = local_to_global_mapping(ii)
        patch_op[ii][ii] = blocks_op[I][I]
        if block_prod:
            patch_prod[ii][ii] = blocks_prod[I][I]
        patch_rhs[ii] = blocks_rhs[I, 0]
        for J in neighbors(I):
            jj = global_to_local_mapping(J)
            if jj >= 0:
                # coupling contribution because nn is inside the patch
                patch_op[ii][jj] = blocks_op[I][J]
                if block_prod:
                    patch_prod[ii][jj] = blocks_prod[I][J]
            else:
                # fake dirichlet contribution because nn is outside the patch
                # seems to be wrong !
                # patch_op[ii][ii] += ops_dirichlet[ss][nn]
                pass

    # if lincomb_outside and block_rhs.parametric:
    #     # porkelei for efficient residual reduction
    #     # change from BlockOperator(LincombOperators) to LincombOperator(BlockOperators)
    #     rhs_operators = []
    #     # this only works for globally defined parameter functionals
    #     # we only take the coefficients of the first one and assert below that this is
    #     # always the same
    #     rhs_coefficients = patch_rhs[0].coefficients
    #     blocks = [np.empty(S_patch, dtype=object)
    #               for _ in range(len(rhs_coefficients))]
    #     for I in range(S_patch):
    #         rhs_lincomb = patch_rhs[I]
    #         # asserts that parameter functionals are globally defined
    #         assert rhs_lincomb.coefficients == rhs_coefficients
    #         for i_, rhs in enumerate(rhs_lincomb.operators):
    #             blocks[i_][I] = rhs
    #     rhs_operators = [BlockColumnOperator(block) for block in blocks]
    #     final_patch_rhs = LincombOperator(rhs_operators, rhs_coefficients)
    # else:
    final_patch_rhs = BlockColumnOperator(patch_rhs)

    # if lincomb_outside and block_op.parametric:
    #     # porkelei for efficient residual reduction
    #     # change from BlockOperator(LincombOperators) to LincombOperator(BlockOperators)
    #     op_operators = []
    #     # this only works for globally defined parameter functionals
    #     # we only take the coefficients of the first one and assert below that this is
    #     # always the same
    #     op_param_coefficients = list(set(
    #         [coef for coef in patch_op[0, 0].coefficients
    #          if isinstance(coef, ParameterFunctional) and coef.parametric]))
    #     op_coefficients = op_param_coefficients + [1.]  # <-- for non parametric parts
    #     blocks = [np.empty((S_patch, S_patch), dtype=object) for _ in range(len(op_coefficients))]
    #     for I in range(S_patch):
    #         for J in range(S_patch):
    #             op_lincomb = patch_op[I, J]
    #             if op_lincomb:     # can be None
    #                 if not op_lincomb.parametric:
    #                     # this is for the coupling parts that are independent of the parameter
    #                     assert len(op_lincomb.operators) == 1 and op_lincomb.coefficients[0] == 1.
    #                     blocks[-1][I, J] = op_lincomb.operators[0]
    #                 else:
    #                     constant_ops = []
    #                     parametric_ops = []
    #                     param_coefficients_in_op = []
    #                     for coef, op in zip(op_lincomb.coefficients, op_lincomb.operators):
    #                         if isinstance(coef, ParameterFunctional) and coef.parametric:
    #                             parametric_ops.append(op)
    #                             param_coefficients_in_op.append(coef)
    #                         else:
    #                             constant_ops.append(op)

    #                     for coef, op in zip(param_coefficients_in_op, parametric_ops):
    #                         for i_, coef_ in enumerate(op_param_coefficients):
    #                             if coef_ == coef:
    #                                 if blocks[i_][I, J] is None:
    #                                     blocks[i_][I, J] = op
    #                                 else:
    #                                     blocks[i_][I, J] += op
    #                                 break   # coefficients are unique

    #                     blocks[-1][I, J] = sum(constant_ops)
    #     op_operators = [BlockOperator(block) for block in blocks]
    #     final_patch_op = LincombOperator(op_operators, op_coefficients)
    # else:
    final_patch_op = BlockOperator(patch_op)

    final_patch_prod = BlockOperator(patch_prod) if block_prod else None
    products = dict(h1=final_patch_prod) if block_prod else None

    patch_model = StationaryModel(operator=final_patch_op, rhs=final_patch_rhs,
                                  products=products)
    return patch_model, local_to_global_mapping, global_to_local_mapping


def construct_element_patches(dd_grid):
    # This is only working with quadrilateral meshes right now !
    # TODO: assert this
    element_patches = []
    for ss in range(dd_grid.num_subdomains):
        nh = {ss}
        nh.update(dd_grid.neighbors(ss))
        for nn in dd_grid.neighbors(ss):
            for nnn in dd_grid.neighbors(nn):
                if nnn not in nh and len(set(dd_grid.neighbors(nnn)).intersection(nh)) == 2:
                    nh.add(nnn)
        element_patches.append(tuple(nh))
    return element_patches


def construct_inner_node_patches(dd_grid):
    # This is only working with quadrilateral meshes right now !
    # TODO: assert this
    domain_ids = np.arange(dd_grid.num_subdomains).reshape(
        (int(np.sqrt(dd_grid.num_subdomains)),) * 2)
    node_patches = {}
    for i in range(domain_ids.shape[0] - 1):
        for j in range(domain_ids.shape[1] - 1):
            node_patches[i * domain_ids.shape[1] + j] = \
                ([domain_ids[i, j], domain_ids[i, j+1],
                  domain_ids[i+1, j], domain_ids[i+1, j+1]])
    return node_patches


def add_element_neighbors(dd_grid, domains):
    new_domains = {}
    for (i, elements) in domains.items():
        elements_and_all_neighbors = []
        for el in elements:
            elements_and_all_neighbors.extend(dd_grid.neighbors(el))
        new_domains[i] = list(np.sort(np.unique(elements_and_all_neighbors)))
    return new_domains


def remove_irrelevant_coupling_from_patch_operator(patch_model, mapping_to_global):
    local_op = patch_model.operator
    subspaces = len(local_op.range.subspaces)
    ops_without_outside_coupling = np.empty((subspaces, subspaces), dtype=object)

    blocks = local_op.blocks
    for i in range(subspaces):
        for j in range(subspaces):
            if not i == j:
                if blocks[i][j]:
                    # the coupling stays
                    ops_without_outside_coupling[i][j] = blocks[i][j]
            else:
                # only the irrelevant couplings need to disappear
                if blocks[i][j]:
                    # only works for one thermal block right now
                    I = mapping_to_global(i)
                    element_patch = [mapping_to_global(l) for l in np.arange(subspaces)]
                    strings = []
                    for J in element_patch:
                        if I < J:
                            strings.append(f'{I}_{J}')
                        else:
                            strings.append(f'{J}_{I}')
                    local_ops, local_coefs = [], []
                    for op, coef in zip(blocks[i][j].operators, blocks[i][j].coefficients):
                        if ('volume' in op.name or 'boundary' in op.name
                           or np.sum(string in op.name for string in strings)):
                            local_ops.append(op)
                            local_coefs.append(coef)

                    ops_without_outside_coupling[i][i] = LincombOperator(local_ops, local_coefs)

    return BlockOperator(ops_without_outside_coupling)


def project_block_operator(operator, range_bases, source_bases):
    # TODO: implement this with ruletables
    # see PR #894 in pymor
    if isinstance(operator, LincombOperator):
        operators = []
        for op in operator.operators:
            operators.append(_project_block_operator(op, range_bases, source_bases))
        return LincombOperator(operators, operator.coefficients)
    else:
        return _project_block_operator(operator, range_bases, source_bases)


def _project_block_operator(operator, range_bases, source_bases):
    if isinstance(operator, IdentityOperator):
        local_projected_op = np.empty((len(range_bases), len(source_bases)), dtype=object)
        # TODO: assert that both bases are the same
        for I in range(len(range_bases)):
            local_projected_op[I][I] = project(IdentityOperator(range_bases[I].space),
                                               range_bases[I], range_bases[I])
    elif isinstance(operator, BlockOperator):
        local_projected_op = np.empty_like(operator.blocks)
        for I in range(len(range_bases)):
            local_basis_I = range_bases[I]
            for J in range(len(source_bases)):
                local_basis_J = source_bases[J]
                if operator.blocks[I][J]:
                    local_projected_op[I][J] = project(operator.blocks[I][J],
                                                       local_basis_I, local_basis_J)
    else:
        raise NotImplementedError
    projected_operator = BlockOperator(local_projected_op)
    return projected_operator


def project_block_rhs(rhs, range_bases):
    # TODO: implement this with ruletables
    # see PR #894 in pymor
    if isinstance(rhs, LincombOperator):
        operators = []
        for op in rhs.operators:
            operators.append(_project_block_rhs(op, range_bases))
        return LincombOperator(operators, rhs.coefficients)
    else:
        return _project_block_rhs(rhs, range_bases)


def _project_block_rhs(rhs, range_bases):
    if isinstance(rhs, VectorOperator):
        rhs_blocks = rhs.array._blocks
        local_projected_rhs = np.empty(len(rhs_blocks), dtype=object)
        for I in range(len(range_bases)):
            rhs_operator = VectorArrayOperator(rhs_blocks[I])
            local_projected_rhs[I] = project(rhs_operator, range_bases[I], None)
    elif isinstance(rhs, BlockColumnOperator):
        local_projected_rhs = np.empty_like(rhs.blocks[:, 0])
        for I in range(len(range_bases)):
            local_projected_rhs[I] = project(rhs.blocks[I, 0], range_bases[I], None)
    else:
        raise NotImplementedError
    projected_rhs = BlockColumnOperator(local_projected_rhs)
    return projected_rhs


class GlobalEllipticEstimator(ImmutableObject):

    def __init__(self, fom):
        self.__auto_init(locals())

    def estimate_error(self, u, mu):
        assert u in self.fom.solution_space
        product = self.fom.products['h1']
        operator = self.fom.operator
        rhs = self.fom.rhs
        riesz_rep = product.apply_inverse(operator.apply(u, mu) - rhs.as_vector(mu))
        return np.sqrt(product.apply2(riesz_rep, riesz_rep))


class EllipticIPLRBEstimator(ImmutableObject):

    def __init__(self, estimator_data, residuals, domains):
        self.__auto_init(locals())

    def estimate_error(self, u_rom, mu):
        indicators = []

        for (domain, (elements, inner_elements, local_inner_elements)), residual in zip(
                self.estimator_data.items(), self.residuals):
            u_in_ed = u_rom.block(elements)

            if isinstance(residual, ResidualReductor):
                # TODO: this here is the non-reduced case where residual is the ResidualReductor
                residual_operator = ResidualOperator(residual.operator, residual.rhs)

                u_in_ed = residual_operator.source.make_array(u_in_ed)
                residual_full = residual_operator.apply(u_in_ed, mu)

                # old approach without product:
                # res_on_node_patch = [residual_full.block(el).norm() for el in local_inner_elements]
                # norm = np.linalg.norm(res_on_node_patch)

                # new approach: cut out only the relevent part. TODO: use concatenation?
                residual_inner_vec = [residual_full.block(i) if i in local_inner_elements
                                      else residual_full.block(i).space.zeros()
                                      for i in range(len(elements))]
                residual_inner_vec = residual.product.source.make_array(residual_inner_vec)
                res = residual.product.apply_inverse(residual_inner_vec)
                norm = np.sqrt(residual.product.apply2(res, res))
            else:
                assert 0
                # this is the reduced case where residual is the reduced residual
                # NOTE: this can not be used, see the comment in _reduce_residual !!
                # norm = residual.apply(u_in_ed, mu=mu).norm()

            indicators.append(norm)

        # distribute indicators to domains TODO: Is this the correct way of distributing this?
        ests = np.zeros(self.domains)
        for associated_domain, ind in zip(sorted(self.estimator_data.keys()), indicators):
            elements = self.estimator_data[associated_domain][0]
            for sd in elements:
                ests[sd] += ind**2
        estimate = np.sqrt(sum(ests))
        return estimate, np.linalg.norm(indicators), indicators

