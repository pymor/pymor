# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.projection import project, project_to_subbasis
from pymor.models.basic import StationaryModel
from pymor.models.saddle_point import SaddlePointModel
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import ProjectionBasedReductor, StationaryLSRBReductor
from pymor.vectorarrays.block import BlockVectorSpace


class StationaryRBStokesReductor(ProjectionBasedReductor):
    """Projection-based reductor for the stationary stokes equation.

    Parameters
    ----------
    fom
        The Stokes |SaddlePointModel| to reduce.
    RB_u
        The basis of the reduced velocity space onto which to project.
        If `None` an empty basis is used.
    RB_p
        The basis of the redcued pressure space onto which to project.
        If `None` an empty basis is used.
    projection_method
        'supremizer_galerkin', 'ls-ls' or 'ls-normal'. Default is 'supremizer_galerkin', which uses
        a supremizer enrichment strategy for the velocity space in order to guarantee
        inf-sup stability. 'ls-ls' solves the least-squares problem directly using a
        least-squares solver. 'ls-normal' solves the least-squares problem via the normal equation.
    u_product
        Inner product |Operator| w.r.t. which `RB_u` is orthonormalized.
        If `None`, the Euclidean inner product is used.
    p_product
        Inner product |Operator| w.r.t. which `RB_p` is orthonormalized.
        If `None`, the Euclidean inner product is used.
    check_orthonormality
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    check_tol
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    """

    def __init__(self, fom, RB_u=None, RB_p=None, projection_method='supremizer_galerkin',
                 u_product=None, p_product=None, check_orthonormality=None, check_tol=None):
        assert isinstance(fom, SaddlePointModel)
        RB_u = fom.solution_space.subspaces[0].empty() if RB_u is None else RB_u
        RB_p = fom.solution_space.subspaces[1].empty() if RB_p is None else RB_p
        assert RB_u in fom.solution_space.subspaces[0]
        assert RB_p in fom.solution_space.subspaces[1]

        self.projection_method = projection_method

        super().__init__(fom, {'RB_u': RB_u, 'RB_p': RB_p}, {'RB_u': u_product, 'RB_p': p_product},
                        check_orthonormality=check_orthonormality, check_tol=check_tol)

    def project_operators(self):
        fom = self.fom
        RB_u = self.bases['RB_u']
        RB_p = self.bases['RB_p']

        u_product = self.products['RB_u']
        p_product = self.products['RB_p']

        if self.projection_method == 'supremizer_galerkin':
            supremizers = self.compute_supremizers()
            self.extend_basis(supremizers, 'RB_u')
            RB_u_enriched = self.bases['RB_u']

            trial_space = BlockVectorSpace((RB_u_enriched.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u_enriched, RB_p))

            projected_operators = {
                'operator':          project(fom.operator, range_basis=V_block, source_basis=V_block),
                'rhs':               project(fom.rhs, range_basis=V_block, source_basis=None),
                'products':          {'u': project(fom.products['u'], RB_u_enriched, RB_u_enriched),
                                      'p': project(fom.products['p'], RB_p, RB_p)},
                'output_functional': project(fom.output_functional, None, V_block)
            }

        elif self.projection_method == 'ls-normal':
            trial_space = BlockVectorSpace((RB_u.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u, RB_p))

            if u_product and p_product:
                mixed_product = BlockDiagonalOperator(blocks=[u_product, p_product])
            elif u_product and p_product is None:
                mixed_product = BlockDiagonalOperator(blocks=[u_product, IdentityOperator(fom.products['p'].range)])
            elif u_product is None and p_product:
                mixed_product = BlockDiagonalOperator(blocks=[IdentityOperator(fom.products['u'].range), p_product])
            else:
                mixed_product = None

            # the helper_fom is required as the StationaryLSRBReductor needs
            # a single product dict for the whole blocked space
            helper_fom = StationaryModel(operator=fom.operator, rhs=fom.rhs, products={'mixed': mixed_product})
            projected_operators = StationaryLSRBReductor(helper_fom, RB=V_block, product=mixed_product,
                                                         use_normal_equations=True).project_operators()

            # project the products again to retrieve the correct blocks
            projected_operators['products'] = {'u': project(fom.products['u'], RB_u, RB_u),
                                               'p': project(fom.products['p'], RB_p, RB_p)}

        elif self.projection_method == 'ls-ls':
            trial_space = BlockVectorSpace((RB_u.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u, RB_p))

            if u_product and p_product:
                mixed_product = BlockDiagonalOperator(blocks=[u_product, p_product])
            elif u_product and p_product is None:
                mixed_product = BlockDiagonalOperator(blocks=[u_product, IdentityOperator(fom.products['p'].range)])
            elif u_product is None and p_product:
                mixed_product = BlockDiagonalOperator(blocks=[IdentityOperator(fom.products['u'].range), p_product])
            else:
                mixed_product = None

            # the helper_fom is required as the StationaryLSRBReductor needs
            # a single product dict for the whole blocked space
            helper_fom = StationaryModel(operator=fom.operator, rhs=fom.rhs, products={'mixed': mixed_product})
            projected_operators = StationaryLSRBReductor(helper_fom, RB=V_block,
                                                         product=mixed_product).project_operators()

            # project the products again to retrieve the correct blocks
            projected_operators['products'] = {'u': project(fom.products['u'], RB_u, RB_u),
                                               'p': project(fom.products['p'], RB_p, RB_p)}

        else:
            raise NotImplementedError

        return projected_operators

    def compute_supremizers(self):
        fom = self.fom
        RB_p = self.bases['RB_p']
        u_product = self.products['RB_u']

        block_pu = fom.operator.blocks[1,0]
        supremizers = fom.solution_space.subspaces[0].empty()

        supremizer_rhs = block_pu.apply_adjoint(RB_p)
        if u_product:
            supremizer_vector = u_product.apply_inverse(supremizer_rhs)
        else:
            supremizer_vector = supremizer_rhs

        supremizers.append(supremizer_vector)

        return supremizers

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim_u = dims['RB_u']
        dim_p = dims['RB_p']

        if self.projection_method == 'supremizer_galerkin':
            dim_u = dims['RB_u_enriched']

        projected_operators = {
            'operator':          project_to_subbasis(rom.operator, dim_u + dim_p, dim_u + dim_p),
            'rhs':               project_to_subbasis(rom.rhs, dim_u + dim_p, None),
            'products':          {'u': project_to_subbasis(rom.products['u'], dim_u, dim_u),
                                  'p': project_to_subbasis(rom.products['p'], dim_p, dim_p)},
            'output_functional': project_to_subbasis(rom.output_functional, None, dim_u + dim_p)
        }

        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        return StationaryModel(error_estimator=error_estimator, **projected_operators)
