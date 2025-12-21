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


class SupremizerGalerkinStokesReductor(ProjectionBasedReductor):
    """Projection-based reductor for the stationary stokes equation.

    Uses supremizer enrichment to stabilize the velocity space, then computes
    a Galerkin projection of the full order model onto the reduced space.

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

    def __init__(self, fom, RB_u=None, RB_p=None, u_product=None, p_product=None,
                 check_orthonormality=None, check_tol=None):
        assert isinstance(fom, SaddlePointModel)
        RB_u = fom.solution_space.subspaces[0].empty() if RB_u is None else RB_u
        RB_p = fom.solution_space.subspaces[1].empty() if RB_p is None else RB_p
        assert RB_u in fom.solution_space.subspaces[0]
        assert RB_p in fom.solution_space.subspaces[1]

        self.u_product = u_product
        self.supremizers = fom.solution_space.subspaces[0].empty()

        super().__init__(fom, {'RB_u': RB_u, 'RB_p': RB_p}, {'RB_u': u_product, 'RB_p': p_product},
                        check_orthonormality=check_orthonormality, check_tol=check_tol)

    def project_operators(self):
        fom = self.fom
        RB_u = self.bases['RB_u']
        RB_p = self.bases['RB_p']

        if len(self.supremizers) < len(RB_p):
            # Enrich velocity basis RB_u with one supremizer per pressure basis vector:
            # len(RB_u) <- len(RB_u) + len(RB_p)
            with self.logger.block(
                'Enriching RB_u with supremizers'
            ):
                new_supremizers = self.compute_supremizers(RB_p, offset=len(self.supremizers))
                self.supremizers.append(new_supremizers, remove_from_other=True)

        RB_u_enriched = cat_arrays([RB_u, self.supremizers])
        RB_u_enriched = gram_schmidt(RB_u_enriched, offset=len(RB_u), product=self.u_product, copy=False)
        self._block_basis = fom.solution_space.make_block_diagonal_array((RB_u_enriched, RB_p))
        block_stationary_reductor = StationaryRBReductor(fom, RB=self._block_basis, check_orthonormality=False)

        return block_stationary_reductor.project_operators()

    def compute_supremizers(self, RB_p, offset=0):
        fom = self.fom
        block_pu = fom.operator.blocks[1,0]

        supremizer_rhs = block_pu.apply_adjoint(RB_p[offset:])
        if fom.u_product:
            supremizer_vector = fom.u_product.apply_inverse(supremizer_rhs)
        else:
            supremizer_vector = supremizer_rhs

        return supremizer_vector

    # overwriting reduce from ProjectionBasedReductor as we currently do not support
    # project_operators_to_subbasis for the SupremizerGalerkinStokesReductor.
    def reduce(self):
        return self._reduce()

    def build_rom(self, projected_operators, error_estimator):
        return StationaryModel(error_estimator=error_estimator, **projected_operators)

    def reconstruct(self, u):
        return self._block_basis.lincomb(u.to_numpy())


class LSRBStokesReductor(StationaryLSRBReductor):
    """Projection-based least-squares reductor for the stationary stokes equation.

    Parameters
    ----------
    fom
        See :class:`SupremizerGalerkinStokesReductor`.
    RB_u
        See :class:`SupremizerGalerkinStokesReductor`.
    RB_p
        See :class:`SupremizerGalerkinStokesReductor`.
    u_product
        See :class:`SupremizerGalerkinStokesReductor`.
    p_product
        See :class:`SupremizerGalerkinStokesReductor`.
    use_normal_equations
        Whether to solve the least-squares problem directly using a least-squares solver
        or via the normal equations.
    check_orthonormality
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    check_tol
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    """

    def __init__(self, fom, RB_u=None, RB_p=None, u_product=None, p_product=None,
                 use_normal_equations=False, check_orthonormality=None, check_tol=None):
        assert isinstance(fom, SaddlePointModel)

        RB = fom.solution_space.make_block_diagonal_array([RB_u, RB_p])
        product = None
        if u_product or p_product:
            blocks = [
                    u_product if u_product else IdentityOperator(fom.solution_space.subspaces[0]),
                    p_product if p_product else IdentityOperator(fom.solution_space.subspaces[1])
                ]
            product = BlockDiagonalOperator(blocks)

        super().__init__(fom, RB=RB, product=product, use_normal_equations=use_normal_equations,
                         check_orthonormality=check_orthonormality, check_tol=check_tol)

    # overwriting reduce from ProjectionBasedReductor as we currently do not support
    # project_operators_to_subbasis for the LSRBStokesReductor.
    def reduce(self):
        return self._reduce()
