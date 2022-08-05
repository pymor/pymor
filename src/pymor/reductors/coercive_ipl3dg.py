import numpy as np
from copy import deepcopy

from pymor.operators.constructions import ZeroOperator, LincombOperator, VectorOperator
from pymor.algorithms.projection import project
from pymor.operators.block import BlockOperator

from pymor.models.basic import StationaryModel
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.reductors.basic import extend_basis
from pymor.algorithms.gram_schmidt import gram_schmidt

class CoerciveIPLD3GRBReductor(CoerciveRBReductor):
    def __init__(self, fom, dd_grid):
        self.fom = fom
        self.solution_space = fom.solution_space
        self.S = self.solution_space.empty().num_blocks
        self._last_rom = None

        self.local_bases = [self.solution_space.empty().block(ss).empty()
                            for ss in range(self.S)]

        neighborhoods = construct_neighborhoods(dd_grid)

        patch_models = []
        patch_mappings_to_global = []
        patch_mappings_to_local = []
        for neighborhood in neighborhoods:
            patch_model, local_to_global_mapping, global_to_local_mapping = construct_patch_model(
                neighborhood, fom.operator, fom.rhs, dd_grid.neighbors)
            patch_models.append(patch_model)
            patch_mappings_to_global.append(local_to_global_mapping)
            patch_mappings_to_local.append(global_to_local_mapping)

        self.neighborhoods = neighborhoods
        self.patch_models = patch_models
        self.patch_mappings_to_global = patch_mappings_to_global
        self.patch_mappings_to_local = patch_mappings_to_local

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
        assert self._last_rom is not None
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
            rhs = patch_model.rhs - a_u_v_as_operator)
        # TODO: think about removing boundary dofs for the rhs
        phi = patch_model_with_correction.solve(mu)
        self.add_local_solutions(I, phi.block(mapping_to_local(I)))
        return phi

    def basis_length(self):
        return [len(self.local_bases[I]) for I in range(self.S)]

    def reduce(self, dims=None):
        assert dims is None, 'we cannot yet reduce to subbases'

        if self._last_rom is None or sum(self.basis_length()) > self._last_rom_dims:
            self._last_rom = self._reduce()
            self._last_rom_dims = sum(self.basis_length())
            # self.reduced_local_basis is required to perform multiple local enrichments
            # after the other
            # TODO: check how to fix this better
            self.reduced_local_bases = deepcopy(self.local_bases)

        return self._last_rom

    def project_operators(self):
        projected_ops_blocks = []
        # this is for BlockOperator(LincombOperators)
        assert isinstance(self.fom.operator, BlockOperator)
        assert not self.fom.rhs.parametric

        local_projected_op = np.empty((self.S, self.S), dtype=object)
        for I in range(self.S):
            for J in range(self.S):
                local_basis_I = self.local_bases[I]
                local_basis_J = self.local_bases[J]
                if self.fom.operator.blocks[I][J]:
                    local_projected_op[I][J] = project(self.fom.operator.blocks[I][J],
                                                       local_basis_I, local_basis_J)
        projected_operator = BlockOperator(local_projected_op)

        local_projected_rhs = np.empty(self.S, dtype=object)
        for I in range(self.S):
            # TODO: find an easier way for this this is currently not possible for parametric rhs
            local_basis = self.local_bases[I]
            rhs_int = project(self.fom.rhs.blocks[I, 0], local_basis, None).matrix[:, 0]
            local_projected_rhs[I] = local_projected_op[I][I].range.make_array(rhs_int)
        projected_rhs = VectorOperator(projected_operator.range.make_array(local_projected_rhs))
        # projected_rhs = BlockOperator(local_projected_rhs)

        projected_operators = {
            'operator':          projected_operator,
            'rhs':               projected_rhs,
            'products':          None,
            'output_functional': None
        }
        return projected_operators

    def assemble_error_estimator(self):
        return None

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

def construct_patch_model(neighborhood, block_op, block_rhs, neighbors):
    def local_to_global_mapping(i):
        return neighborhood[i]
    def global_to_local_mapping(i):
        for i_, j in enumerate(neighborhood):
            if j == i:
                return i_
        return -1

    S_patch = len(neighborhood)
    patch_op = np.empty((S_patch, S_patch), dtype=object)
    patch_rhs = np.empty(S_patch, dtype=object)
    blocks_op = block_op.blocks
    blocks_rhs = block_rhs.blocks
    for ii in range(S_patch):
        I = local_to_global_mapping(ii)
        patch_op[ii][ii] = blocks_op[I][I]
        patch_rhs[ii] = blocks_rhs[I, 0].array
        for J in neighbors(I):
            jj = global_to_local_mapping(J)
            if jj >= 0:
                # coupling contribution because nn is inside the patch
                patch_op[ii][jj] = blocks_op[I][J]
            else:
                # fake dirichlet contribution because nn is outside the patch
                # patch_op[ii][ii] += ops_dirichlet[ss][nn]
                pass

    final_patch_op = BlockOperator(patch_op)
    final_patch_rhs = VectorOperator(final_patch_op.range.make_array(patch_rhs))

    patch_model = StationaryModel(final_patch_op, final_patch_rhs)
    return patch_model, local_to_global_mapping, global_to_local_mapping

def construct_neighborhoods(dd_grid):
    # This is only working with quadrilateral meshes right now !
    # TODO: assert this
    neighborhoods = []
    for ss in range(dd_grid.num_subdomains):
        nh = {ss}
        nh.update(dd_grid.neighbors(ss))
        for nn in dd_grid.neighbors(ss):
            for nnn in dd_grid.neighbors(nn):
                if nnn not in nh and len(set(dd_grid.neighbors(nnn)).intersection(nh)) == 2:
                    nh.add(nnn)
        neighborhoods.append(tuple(nh))
    return neighborhoods

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
                    neighborhood = [mapping_to_global(l) for l in np.arange(subspaces)]
                    strings = []
                    for J in neighborhood:
                        if I < J:
                            strings.append(f'{I}_{J}')
                        else:
                            strings.append(f'{J}_{I}')
                    local_ops, local_coefs = [], []
                    for op, coef in zip(blocks[i][j].operators, blocks[i][j].coefficients):
                        if ('volume' in op.name or 'boundary' in op.name or
                                     np.sum(string in op.name for string in strings)):
                            local_ops.append(op)
                            local_coefs.append(coef)

                    ops_without_outside_coupling[i][i] = LincombOperator(local_ops, local_coefs)

    return BlockOperator(ops_without_outside_coupling)
