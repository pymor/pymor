# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.projection import project_to_subbasis
from pymor.models.basic import StationaryModel
from pymor.models.saddle_point import SaddlePointModel
from pymor.reductors.basic import ProjectionBasedReductor, StationaryLSRBReductor, StationaryRBReductor


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

        if self.projection_method == 'supremizer_galerkin':
            supremizers = self.compute_supremizers()
            self.extend_basis(supremizers, 'RB_u')
            V_block = fom.solution_space.make_block_diagonal_array((self.bases['RB_u'], RB_p))
            projected_operators = StationaryRBReductor(fom, RB=V_block, check_orthonormality=False).project_operators()

        elif self.projection_method == 'ls-normal' or self.projection_method == 'ls-ls':
            V_block = fom.solution_space.make_block_diagonal_array((RB_u, RB_p))
            mixed_product = fom.products['mixed'] if fom.products else None

            if self.projection_method == 'ls-normal':
                projected_operators = StationaryLSRBReductor(fom, RB=V_block, product=mixed_product,
                                                             use_normal_equations=True,
                                                             check_orthonormality=False).project_operators()
            else:
                projected_operators = StationaryLSRBReductor(fom, RB=V_block, product=mixed_product,
                                                             check_orthonormality=False).project_operators()

        else:
            raise NotImplementedError

        return projected_operators

    def compute_supremizers(self):
        fom = self.fom
        RB_p = self.bases['RB_p']
        u_product = fom.u_product

        block_pu = fom.operator.blocks[1,0]
        supremizers = fom.solution_space.subspaces[0].empty()

        supremizer_rhs = block_pu.apply_adjoint(RB_p)
        if u_product:
            supremizer_vector = u_product.apply_inverse(supremizer_rhs)
        else:
            supremizer_vector = supremizer_rhs

        norm = supremizer_vector.norm()
        supremizer_vector.scal(1 / norm)
        supremizers.append(supremizer_vector)

        return supremizers

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim_u = dims['RB_u']
        dim_p = dims['RB_p']
        dim_trial = dim_u + dim_p

        if self.projection_method == 'supremizer_galerkin':
            dim_u_enriched = dims['RB_u_enriched']
            dim_trial = dim_u_enriched + dim_p

        if self.projection_method == 'supremizer_galerkin' or self.projection_method == 'ls-normal':
            # Square system: both dimensions are the same
            projected_operators = {
                'operator':          project_to_subbasis(rom.operator, dim_trial, dim_trial),
                'rhs':               project_to_subbasis(rom.rhs, dim_trial, None),
                'output_functional': project_to_subbasis(rom.output_functional, None, dim_trial)
            }

        else:
            # Rectangular system: test space (range) is larger than trial space (source)
            dim_test = rom.operator.range.dim
            projected_operators = {
                'operator':          project_to_subbasis(rom.operator, dim_test, dim_trial),
                'rhs':               project_to_subbasis(rom.rhs, dim_test, None),
                'output_functional': project_to_subbasis(rom.output_functional, None, dim_trial)
            }
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        return StationaryModel(error_estimator=error_estimator, **projected_operators)
