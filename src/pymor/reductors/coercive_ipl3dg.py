import numpy as np
from copy import deepcopy

from pymor.algorithms.projection import project
from pymor.core.base import ImmutableObject
from pymor.core.defaults import set_defaults
from pymor.core.exceptions import ExtensionError
from pymor.models.basic import StationaryModel
from pymor.operators.block import BlockOperator, BlockColumnOperator
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.operators.constructions import VectorArrayOperator, IdentityOperator
from pymor.parameters.functionals import ParameterFunctional
from pymor.reductors.basic import extend_basis
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.residual import ResidualReductor, ResidualOperator


class CoerciveIPLD3GRBReductor(CoerciveRBReductor):
    def __init__(self, fom, dd_grid, local_bases=None, product=None,
                 localized_estimator=False, reductor_type='residual',
                 unassembled_product=None, fake_dirichlet_ops=None):
        """
            TBC

            reductor_type:
                'residual'      : Reduction of the residual based on ResidualReductor.
                'simple'        : Reduction of the residual based on SimpleCoerciveRBReductor.
                'non_assembled' : No reduction.
        """
        self.__auto_init(locals())

        if localized_estimator:
            assert unassembled_product is not None
            assert fake_dirichlet_ops is not None

        self.solution_space = fom.solution_space
        self.S = self.solution_space.empty().num_blocks
        self._last_rom = None

        # TODO: assertions for local_bases
        self.local_bases = local_bases or [self.solution_space.empty().block(I).empty()
                                           for I in range(self.S)]
        self.local_products = [None if not product else product.blocks[I, I]
                               for I in range(self.S)]

        # element patches for online enrichment
        self.element_patches = construct_element_patches(dd_grid)

        self.patch_models = []
        self.patch_mappings_to_global = []
        self.patch_mappings_to_local = []
        for element_patch in self.element_patches:
            patch_model, local_to_global, global_to_local = construct_local_model(
                element_patch, fom.operator, fom.rhs, dd_grid.neighbors,
                ops_dirichlet=self.fake_dirichlet_ops)
            self.patch_models.append(patch_model)
            self.patch_mappings_to_global.append(local_to_global)
            self.patch_mappings_to_local.append(global_to_local)

        if localized_estimator:
            # node patches for estimation
            # the estimator domains are node patches
            inner_node_patches = construct_inner_node_patches(dd_grid)
            # estimator_domains = inner_node_patches
            estimator_domains = add_element_neighbors(dd_grid, inner_node_patches)

            self.estimator_data = {}
            self.inner_products = {}
            self.bases_in_local_domains = {}

            self.estimator_mappings_to_global = []
            self.estimator_mappings_to_local = []
            self.local_support_mappings_to_local = []

            self.local_residuals = []

            for associated_element, elements in estimator_domains.items():
                # construct estimator model on the stencil for residual computation
                local_model, local_to_global, global_to_local = construct_local_model(
                    elements, fom.operator, fom.rhs, dd_grid.neighbors,
                    block_prod=self.unassembled_product, lincomb_outside=True,
                    ops_dirichlet=self.fake_dirichlet_ops)
                node_elements = inner_node_patches[associated_element]
                self.estimator_mappings_to_global.append(local_to_global)
                self.estimator_mappings_to_local.append(global_to_local)
                # TODO: vectorize the following
                local_node_elements = [global_to_local(el) for el in node_elements]
                self.estimator_data[associated_element] = (elements, node_elements, local_node_elements)

                # gather basis
                bases_in_local_domain = [self.local_bases[el] for el in elements]
                self.bases_in_local_domains[associated_element] = bases_in_local_domain

                # todo: we need sparse blockvectorarrays here !!
                basis = local_model.solution_space.make_block_diagonal_array(bases_in_local_domain)
                local_product = local_model.products['product']

                # construct local support model to extract the local inner product
                inner_model, _, global_to_local = construct_local_model(
                    inner_node_patches[associated_element],
                    fom.operator, fom.rhs, dd_grid.neighbors,
                    block_prod=self.unassembled_product, lincomb_outside=True,
                    ops_dirichlet=self.fake_dirichlet_ops)
                self.local_support_mappings_to_local.append(global_to_local)
                inner_product = inner_model.products['product']

                # this is the "product" which is used for the localized estimation.
                # In fact, it is not a product on the larger stencil but only on the local support.
                # Currently, this implies that the SimpleCoerciveRBReductor does not work as it should.
                # However, in the ResidualReductor, the product cancels out.
                restricted_product = RestrictedProductOperator(local_product,
                                                               inner_product,
                                                               local_node_elements)
                if reductor_type == 'residual' or reductor_type == 'non_assembled':
                    residual_reductor = ResidualReductor(basis,
                                                         local_model.operator,
                                                         local_model.rhs,
                                                         product=restricted_product,
                                                         riesz_representatives=True)
                elif reductor_type == 'simple':
                    # TODO: fix this case and make it work with the restricted product
                    residual_reductor = SimpleCoerciveRBReductor(local_model, basis,
                                                                 product=restricted_product,
                                                                 check_orthonormality=False)
                else:
                    assert 0, f'reductor type {reductor_type} not known'

                self.inner_products[associated_element] = inner_product
                self.local_residuals.append(residual_reductor)

    def add_partition_of_unity(self):
        for I in range(self.S):
            # TODO: find the correct partition of unity here
            pass

    def extend_bases_with_global_solution(self, us):
        assert us in self.fom.solution_space
        for I in range(self.S):
            u_block = us.block(I)
            self.extend_basis_locally(I, u_block)

    def extend_basis_locally(self, I, u):
        try:
            extend_basis(u, self.local_bases[I], product=self.local_products[I])
            # TODO: can make use of super.extend_basis ?
        except ExtensionError:
            self.logger.warn('Extension failed')

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
        self.extend_basis_locally(I, phi.block(mapping_to_local(I)))
        return phi

    def basis_length(self):
        return [len(self.local_bases[I]) for I in range(self.S)]

    def reduce(self, dims=None):
        assert dims is None, 'Cannot reduce to subbases, yet'

        if self._last_rom is None or sum(self.basis_length()) > self._last_rom_dims:
            self.reduced_residuals = self._reduce_residuals()
            self._last_rom = self._reduce()
            self._last_rom_dims = sum(self.basis_length())
            # self.reduced_local_basis is required to perform multiple local enrichments
            # TODO: check how to make this better
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
        reduced_residuals = []
        if self.localized_estimator:
            for (_, bases_local), residual in zip(self.bases_in_local_domains.items(),
                                                  self.local_residuals):
                if self.reductor_type == 'residual' or self.reductor_type == 'non_assembled':
                    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.check": False})
                    basis = residual.operator.source.make_block_diagonal_array(bases_local)
                    residual_reductor = ResidualReductor(basis,
                                                         residual.operator,
                                                         residual.rhs,
                                                         product=residual.product,
                                                         riesz_representatives=True)
                elif self.reductor_type == 'simple':
                    set_defaults({'pymor.operators.constructions.induced_norm.raise_negative': False})
                    basis = residual.fom.solution_space.make_block_diagonal_array(bases_local)
                    residual_reductor = SimpleCoerciveRBReductor(residual.fom, basis,
                                                                 product=residual.products['RB'],
                                                                 check_orthonormality=False)

                if self.reductor_type == 'residual' or self.reductor_type == 'simple':
                    reduced_residuals.append(residual_reductor.reduce())
                else:
                    reduced_residuals.append(residual_reductor)

        # NOTE: the above requires a reductor that has a changeable basis, which only makes sense
        # if the BlockVectorArray allows for a sparse format.
        # Currently, we do not use any data from before and reduce from scratch !
        # TODO: implement hierarchy.

        return reduced_residuals

    def assemble_error_estimator(self):
        if self.localized_estimator:
            reduced_support_products = self.inner_products
            estimator = EllipticIPLRBEstimator(self.estimator_data,
                                               self.reduced_residuals,
                                               reduced_support_products,
                                               self.S,
                                               self.reconstruct)
        else:
            estimator = GlobalEllipticEstimator(self.fom, self.product, self.reconstruct)
        return estimator

    def reduce_to_subbasis(self, dims):
        raise NotImplementedError

    def reconstruct(self, u_rom):
        u_ = []
        for I in range(self.S):
            basis = self.reduced_local_bases[I]
            u_I = u_rom.block(I)
            u_.append(basis.lincomb(u_I.to_numpy()))
        return self.fom.solution_space.make_array(u_)

    def from_patch_to_global(self, I, u_patch, patch_type='enrichment'):
        # a function that construct a globally defined u from a patch u
        # only here for visualization purposes. not relevant for reductor
        if patch_type == 'enrichment':
            mapping = self.patch_mappings_to_local
        elif patch_type == 'estimation':
            mapping = self.estimator_mappings_to_local
        elif patch_type == 'local_support':
            mapping = self.local_support_mappings_to_local
        else:
            assert 0, f'patch_type = {patch_type} not known'

        u_global = []
        for i in range(self.S):
            i_loc = mapping[I](i)
            if i_loc >= 0 and u_patch.block(i_loc).to_numpy().sum() != 0:
                u_global.append(u_patch.block(i_loc))
            else:
                # the multiplication with 1e-4 is only there for visualization
                # purposes. Otherwise, the colorbar gets scaled up until 1.
                # which is bad
                # TODO: fix colorscale for the plot
                u_global.append(self.solution_space.subspaces[i].ones()*1e-6)
        return self.solution_space.make_array(u_global)


def construct_local_model(local_elements, block_op, block_rhs, neighbors, block_prod=None,
                          lincomb_outside=False, ops_dirichlet=None):
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

        if block_prod:
            # without couplings
            element_patch = [local_to_global_mapping(l) for l in np.arange(S_patch)]
            strings = []
            for J in element_patch:
                if I < J:
                    strings.append(f'{I}_{J}')
                else:
                    strings.append(f'{J}_{I}')
            local_ops, local_coefs = [], []
            # TODO: fix this !! 
            assert isinstance(blocks_op[I][I], LincombOperator), 'not implemented! '
            for op, coef in zip(blocks_op[I][I].operators, blocks_op[I][I].coefficients):
                if ('volume' in op.name or 'boundary' in op.name
                   or np.sum(string in op.name for string in strings)):
                    local_ops.append(op)
                    local_coefs.append(coef)
            patch_op[ii][ii] = LincombOperator(local_ops, local_coefs)
        else:
            # with all entries
            patch_op[ii][ii] = blocks_op[I][I]

        if block_prod:
            # add h1 semi volume part
            for op in blocks_prod[I][I].operators:
                if 'h1' in op.name:
                    patch_prod[ii][ii] = op
            # add boundary part part
            for op in blocks_prod[I][I].operators:
                if 'boundary' in op.name:
                    patch_prod[ii][ii] += op
            # add coupling parts of the inner domain
            # TODO: check whether this is correct or whether all couplings should be used!
            for J in neighbors(I):
                jj = global_to_local_mapping(J)
                if jj >= 0:
                    # J is inside the patch ! search for the operator to add it
                    for op in blocks_prod[I][I].operators:
                        if str(f'_{J}') in op.name:
                            patch_prod[ii][ii] += op
            # print(patch_prod[ii][ii].operators)
            # patch_prod[ii][ii] = sum(blocks_prod[I][I].operators)

        patch_rhs[ii] = blocks_rhs[I, 0]
        for J in neighbors(I):
            jj = global_to_local_mapping(J)
            if jj >= 0:
                # coupling contribution because J is inside the patch
                patch_op[ii][jj] = blocks_op[I][J]
                if block_prod:
                    patch_prod[ii][jj] = blocks_prod[I][J]
            else:
                # fake dirichlet contribution because J is outside the patch
                if block_prod:
                    for op in ops_dirichlet.blocks[I][J].operators:
                        if 'constant' in op.name:
                            # the fake dirichlet operators are also assembled with
                            # the parametric parts (might be wrong)
                            patch_prod[ii][ii] += op
                # patch_op[ii][ii] += ops_dirichlet.blocks[I][J]

    if lincomb_outside and block_rhs.parametric:
        # porkelei for efficient residual reduction
        # change from BlockOperator(LincombOperators) to LincombOperator(BlockOperators)
        rhs_operators = []
        # this only works for globally defined parameter functionals
        # we only take the coefficients of the first one and assert below that this is
        # always the same
        rhs_coefficients = patch_rhs[0].coefficients
        blocks = [np.empty(S_patch, dtype=object)
                  for _ in range(len(rhs_coefficients))]
        for I in range(S_patch):
            rhs_lincomb = patch_rhs[I]
            # asserts that parameter functionals are globally defined
            assert rhs_lincomb.coefficients == rhs_coefficients
            for i_, rhs in enumerate(rhs_lincomb.operators):
                blocks[i_][I] = rhs
        rhs_operators = [BlockColumnOperator(block) for block in blocks]
        final_patch_rhs = LincombOperator(rhs_operators, rhs_coefficients)
    else:
        final_patch_rhs = BlockColumnOperator(patch_rhs)

    if lincomb_outside and block_op.parametric:
        # porkelei for efficient residual reduction
        # change from BlockOperator(LincombOperators) to LincombOperator(BlockOperators)
        op_operators = []
        # this only works for globally defined parameter functionals
        # we only take the coefficients of the first one and assert below that this is
        # always the same
        op_param_coefficients = list(set(
            [coef for coef in patch_op[0, 0].coefficients
             if isinstance(coef, ParameterFunctional) and coef.parametric]))
        op_coefficients = op_param_coefficients + [1.]  # <-- for non parametric parts
        blocks = [np.empty((S_patch, S_patch), dtype=object) for _ in range(len(op_coefficients))]
        for I in range(S_patch):
            for J in range(S_patch):
                op_lincomb = patch_op[I, J]
                if op_lincomb:     # can be None
                    if not op_lincomb.parametric:
                        # this is for the coupling parts that are independent of the parameter
                        assert len(op_lincomb.operators) == 1 and op_lincomb.coefficients[0] == 1.
                        blocks[-1][I, J] = op_lincomb.operators[0]
                    else:
                        constant_ops = []
                        parametric_ops = []
                        param_coefficients_in_op = []
                        for coef, op in zip(op_lincomb.coefficients, op_lincomb.operators):
                            if isinstance(coef, ParameterFunctional) and coef.parametric:
                                parametric_ops.append(op)
                                param_coefficients_in_op.append(coef)
                            else:
                                constant_ops.append(op)

                        for coef, op in zip(param_coefficients_in_op, parametric_ops):
                            for i_, coef_ in enumerate(op_param_coefficients):
                                if coef_ == coef:
                                    if blocks[i_][I, J] is None:
                                        blocks[i_][I, J] = op
                                    else:
                                        blocks[i_][I, J] += op
                                    break   # coefficients are unique

                        blocks[-1][I, J] = sum(constant_ops)
        op_operators = [BlockOperator(block) for block in blocks]
        final_patch_op = LincombOperator(op_operators, op_coefficients)
    else:
        final_patch_op = BlockOperator(patch_op)

    final_patch_prod = BlockOperator(patch_prod) if block_prod else None
    products = dict(product=final_patch_prod) if block_prod else None

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
    local_projected_op = np.empty((len(range_bases), len(source_bases)), dtype=object)
    if isinstance(operator, IdentityOperator):
        # TODO: assert that both bases are the same
        for I in range(len(range_bases)):
            local_projected_op[I][I] = project(IdentityOperator(range_bases[I].space),
                                               range_bases[I], range_bases[I])
    elif isinstance(operator, BlockOperator):
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
    local_projected_rhs = np.empty(len(range_bases), dtype=object)
    if isinstance(rhs, VectorOperator):
        rhs_blocks = rhs.array._blocks
        for I in range(len(range_bases)):
            rhs_operator = VectorArrayOperator(rhs_blocks[I])
            local_projected_rhs[I] = project(rhs_operator, range_bases[I], None)
    elif isinstance(rhs, BlockColumnOperator):
        for I in range(len(range_bases)):
            local_projected_rhs[I] = project(rhs.blocks[I, 0], range_bases[I], None)
    else:
        raise NotImplementedError
    projected_rhs = BlockColumnOperator(local_projected_rhs)
    return projected_rhs


class GlobalEllipticEstimator(ImmutableObject):

    def __init__(self, fom, product, reconstruct):
        self.__auto_init(locals())

    def estimate_error(self, u, mu, m):
        u = self.reconstruct(u)
        assert u in self.fom.solution_space
        product = self.product
        operator = self.fom.operator
        rhs = self.fom.rhs
        print('WARNING: globalized method')
        riesz_rep = product.apply_inverse(operator.apply(u, mu) - rhs.as_vector(mu))
        return np.sqrt(product.apply2(riesz_rep, riesz_rep))


class RestrictedProductOperator(BlockOperator):
    def __init__(self, product, inner_product, inner_blocks):
        self.__auto_init(locals())
        assert product.range == product.source
        # TODO: get rid of to_dense() neccesity
        self.inner_product = inner_product.to_dense()
        super().__init__(product.to_dense().blocks)

    def global_to_local(self, i):
        for i_, j in enumerate(self.inner_blocks):
            if j == i:
                return i_
        return -1

    def apply_inverse(self, V, mu=None):
        assert V in self.range
        V_on_inner = [V.block(i) for i in self.inner_blocks]
        V_on_inner = self.inner_product.range.make_array(V_on_inner)
        V_inverse = self.inner_product.apply_inverse(V_on_inner)
        V_inverse_on_outer = [V_inverse.block(self.global_to_local(i))
                              if i in self.inner_blocks else V.block(i).zeros(len(V))
                              for i in range(V.num_blocks)]
        return self.source.make_array(V_inverse_on_outer)

    def apply2(self, V, U, mu=None):
        assert U in self.source
        assert V in self.range

        U_on_inner = [U.block(i) for i in self.inner_blocks]
        U_on_inner = self.inner_product.source.make_array(U_on_inner)

        V_on_inner = [V.block(i) for i in self.inner_blocks]
        V_on_inner = self.inner_product.range.make_array(V_on_inner)

        return self.inner_product.apply2(V_on_inner, U_on_inner)

    @property
    def H(self):
        raise NotImplementedError

    def apply(self, U, mu=None):
        assert U in self.source

        U_on_inner = [U.block(i) for i in self.inner_blocks]
        U_on_inner = self.inner_product.source.make_array(U_on_inner)
        V = self.inner_product.apply(U_on_inner, mu=mu)

        V_inner_on_outer = [V.block(self.global_to_local(i))
                            if i in self.inner_blocks else U.block(i).zeros(len(U))
                            for i in range(U.num_blocks)]

        return self.range.make_array(V_inner_on_outer)

    def apply_adjoint(self, V, mu=None):
        raise NotImplementedError

    def assemble(self, mu=None):
        raise NotImplementedError

    def as_range_array(self, mu=None):
        raise NotImplementedError

    def as_source_array(self, mu=None):
        raise NotImplementedError

    def d_mu(self, parameter, index=0):
        raise NotImplementedError

    def only_inner(self, U, mu=None):
        # convenience method for restricting U to the inner part. Not connected to the product
        # but useful in SimpleCoerciveRBReductor debugging.
        U_inner_on_outer = [U.block(self.global_to_local(i))
                            if i in self.inner_blocks else U.block(i).zeros(len(U))
                            for i in range(U.num_blocks)]
        return self.range.make_array(U_inner_on_outer)


class EllipticIPLRBEstimator(ImmutableObject):

    def __init__(self, estimator_data, residuals, support_products, domains, reconstruct):
        self.__auto_init(locals())

    def estimate_error(self, u_rom, mu, m=None):
        indicators = []
        residuals = []

        if isinstance(self.residuals[0], ResidualReductor):
            u_rom_copy = u_rom.copy()
            u_rom = self.reconstruct(u_rom)
        for (domain, (elements, inner_elements, local_inner_elements)), residual in zip(
                self.estimator_data.items(), self.residuals):
            u_in_ed = u_rom.block(elements)

            if isinstance(residual, ResidualReductor):
                # this is the non-reduced case where residual is the ResidualReductor
                residual_operator = ResidualOperator(residual.operator, residual.rhs)
                u_in_ed = residual_operator.source.make_array(u_in_ed)
                residual_full = residual_operator.apply(u_in_ed, mu)

                restricted_product = residual.product
                riesz_inner = restricted_product.apply_inverse(residual_full)
                norm = np.sqrt(restricted_product.apply2(riesz_inner, riesz_inner))

                residuals.append((residual_full, riesz_inner))

                ##### DEBUGGING CODE FOR FIXING SimpleCoerciveRBReductor

                # # the same code as above from RB tutorial
                # rieszes = restricted_product.source.empty()
                # for b in residual.RB:
                #     riesz_a = restricted_product.apply_inverse(residual.operator.apply(b, mu))
                #     rieszes.append(riesz_a)
                # riesz_f = restricted_product.apply_inverse(residual.rhs.as_vector(mu))
                # gram_matrix_rr_op = NumpyMatrixOperator(restricted_product.apply2(rieszes, rieszes))
                # gram_matrix_rf_op = NumpyMatrixOperator(restricted_product.apply2(rieszes, riesz_f))
                # G_f = restricted_product.apply2(riesz_f, riesz_f)

                # u_in_ed = u_rom_copy.block(elements)
                # u_in_ed_unblocked = np.array([u.to_numpy() for u in u_in_ed]).flatten()
                # u_in_ed_unblocked = gram_matrix_rr_op.source.from_numpy(u_in_ed_unblocked)
                # norm_ = np.sqrt(gram_matrix_rr_op.apply2(u_in_ed_unblocked, u_in_ed_unblocked)[0][0] \
                #                 - 2 * gram_matrix_rf_op.apply_adjoint(u_in_ed_unblocked).to_numpy()[0][0] + G_f)

                # # the same code as above from Quarteroni book
                # rieszes = restricted_product.source.empty()
                # Us = restricted_product.source.empty()
                # for b in residual.RB:
                #     U = -residual.operator.apply(b, mu)
                #     riesz_a = restricted_product.apply_inverse(U)
                #     Us.append(U)
                #     rieszes.append(riesz_a)
                # f = residual.rhs.as_vector(mu)
                # riesz_f = restricted_product.apply_inverse(f)

                # R_RR = riesz_f.inner(f)
                # R_RO = riesz_f.inner(Us)
                # R_OO = rieszes.inner(Us)

                # estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
                # estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR
                # estimator_matrix[len(R_RR):, len(R_RR):] = R_OO
                # estimator_matrix[:len(R_RR), len(R_RR):] = R_RO
                # estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T

                # estimator_matrix = NumpyMatrixOperator(estimator_matrix)

                # u_in_ed = u_rom_copy.block(elements)
                # u_in_ed_unblocked = np.array([u.to_numpy() for u in u_in_ed]).flatten()
                # C = np.append([1.], u_in_ed_unblocked)
                # C = estimator_matrix.source.from_numpy(C)
                # norm__ = np.sqrt(estimator_matrix.apply2(C, C))

                # print(norm, norm_, norm__)

            elif isinstance(residual, ResidualOperator):
                # this is the reduced case with improved accuracy
                u_in_ed_unblocked = np.array([u.to_numpy() for u in u_in_ed]).flatten()
                u_in_ed_unblocked = residual.source.from_numpy(u_in_ed_unblocked)
                norm = residual.apply(u_in_ed_unblocked, mu=mu).norm()
            elif isinstance(residual, StationaryModel):
                # this is the case with SimpleCoerciveReductor
                u_in_ed_unblocked = np.array([u.to_numpy() for u in u_in_ed]).flatten()
                u_in_ed_unblocked = residual.solution_space.from_numpy(u_in_ed_unblocked)
                norm = residual.error_estimator.estimate_error(u_in_ed_unblocked, mu=mu,
                                                               m=residual)
                # TODO: It seems that something is wrong with the estimation here because there are nans coming in...
                #       resolve this !
                import math
                if math.isnan(norm[0]):
                    print('WARNING: catched a nan!')
                    norm[0] = 0
                # NOTE: the following can not be used because we need to use u_in_ed for the coefficients
                # norm = residual.estimate_error(mu=mu)

            indicators.append(norm)

        # distribute indicators to domains
        # TODO: Is this the correct way of distributing this?
        ests = np.zeros(self.domains)
        for associated_domain, ind in zip(sorted(self.estimator_data.keys()), indicators):
            elements = self.estimator_data[associated_domain][0]
            for sd in elements:
                ests[sd] += ind**2
        estimate = np.sqrt(sum(ests))
        # return estimate
        return estimate, ests, indicators, residuals

