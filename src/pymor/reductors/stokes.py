# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.models.basic import StationaryModel
from pymor.models.saddle_point import SaddlePointModel
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import ProjectionBasedReductor, StationaryLSRBReductor, StationaryRBReductor
from pymor.vectorarrays.constructions import cat_arrays


class StationaryRBStokesReductor(ProjectionBasedReductor):
    """Projection-based reductor for the stationary stokes equation.

    Parameters
    ----------
    fom
        The Stokes |SaddlePointModel| to reduce.
    RB_u
        The basis of the reduced velocity space onto which to project.
        If `None`, an empty basis is used.
    RB_p
        The basis of the redcued pressure space onto which to project.
        If `None`, an empty basis is used.
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
        self.u_product = u_product
        self.supremizers = fom.solution_space.subspaces[0].empty()

        if u_product or p_product:
            blocks = [
                u_product if u_product else IdentityOperator(fom.solution_space.subspace[0].dim),
                p_product if p_product else IdentityOperator(fom.solution_space.subspaces[1].dim)
            ]
            self.mixed_product = BlockDiagonalOperator(blocks=blocks)
        else:
            self.mixed_product = None

        super().__init__(fom, {'RB_u': RB_u, 'RB_p': RB_p}, {'RB_u': u_product, 'RB_p': p_product},
                        check_orthonormality=check_orthonormality, check_tol=check_tol)

    def _project_operators(self, dims):
        fom = self.fom
        dim_u = dims['RB_u']
        dim_p = dims['RB_p']
        RB_u = self.bases['RB_u'][:dim_u]
        RB_p = self.bases['RB_p'][:dim_p]

        V_block = fom.solution_space.make_block_diagonal_array((RB_u, RB_p))
        mixed_product = self.mixed_product

        if self.projection_method == 'supremizer_galerkin':
            if len(self.supremizers) < len(RB_p):
                # Enrich velocity basis RB_u with one supremizer per pressure basis vector:
                # len(RB_u) <- len(RB_u) + len(RB_p) - len(self.supremizers)
                with self.logger.block(
                    'Enriching RB_u with supremizers'
                ):
                    supremizers = self.compute_supremizers(RB_p, offset=len(self.supremizers))
                    RB_u_enriched = cat_arrays([RB_u, self.supremizers, supremizers])
                    RB_u_enriched = gram_schmidt(RB_u_enriched, offset=len(RB_u)+len(self.supremizers),
                                                 product=self.u_product, copy=False, check=False)
                    self.supremizers = supremizers

            elif len(self.supremizers) > len(RB_p):
                with self.logger.block(
                    'Removing supremizer corresponding to pressure basis vector(s)'
                ):
                    self.supremizers = self.supremizers[:len(RB_p)]
                    RB_u_enriched = cat_arrays([RB_u, self.supremizers])
            else:
                RB_u_enriched = cat_arrays([RB_u, self.supremizers])

            # build V_block again as it changed due to supremizer enrichment
            V_block = fom.solution_space.make_block_diagonal_array((RB_u_enriched, RB_p))
            block_stationary_reductor = StationaryRBReductor(fom, RB=V_block, check_orthonormality=False)

        elif self.projection_method == 'ls-normal':
            block_stationary_reductor = StationaryLSRBReductor(fom, RB=V_block, product=mixed_product,
                                                               use_normal_equations=True, check_orthonormality=False)

        elif self.projection_method == 'ls-ls':
            block_stationary_reductor = StationaryLSRBReductor(fom, RB=V_block, product=mixed_product,
                                                               check_orthonormality=False)
        else:
            raise NotImplementedError

        projected_operators = block_stationary_reductor.project_operators()
        self._block_basis = V_block
        return projected_operators

    def compute_supremizers(self, RB_p, offset=0):
        fom = self.fom
        u_product = fom.u_product
        block_pu = fom.operator.blocks[1,0]

        supremizer_rhs = block_pu.apply_adjoint(RB_p[offset:])
        if u_product:
            supremizer_vector = u_product.apply_inverse(supremizer_rhs)
        else:
            supremizer_vector = supremizer_rhs

        norm = supremizer_vector.norm()
        supremizer_vector.scal(1 / norm)

        return supremizer_vector

    def project_operators(self):
        RB_p = self.bases['RB_p']
        RB_u = self.bases['RB_u']
        dims = {'RB_u': len(RB_u), 'RB_p': len(RB_p)}
        return self._project_operators(dims)

    def project_operators_to_subbasis(self, dims):
        return self._project_operators(dims)

    def build_rom(self, projected_operators, error_estimator):
        return StationaryModel(error_estimator=error_estimator, **projected_operators)

    def reconstruct(self, u):
        return self._block_basis[:u.dim].lincomb(u.to_numpy())
