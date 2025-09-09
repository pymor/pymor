# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.reductors.basic import ProjectionBasedReductor
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import InverseOperator, IdentityOperator, AdjointOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.operators.block import BlockDiagonalOperator
from pymor.vectorarrays.constructions import cat_arrays
from pymor.algorithms.gram_schmidt import gram_schmidt


class StationaryRBStokesReductor(ProjectionBasedReductor): 
    """Galerkin or least-sqaures projection of a |StationaryModel| for the stationary stokes equation.

    Parameters
    ----------
    fom
        The Stokes full order |Model| to reduce.
    RB_u
        The basis of the reduced velocity space onto which to project.
        If `None` an empty basis is used.
    RB_p
        The basis of the redcued pressure space onto which to project. 
        If `None` an empty basis is used.
    projection_method 
        'Galerkin', 'ls-ls' or 'ls-normal'. Default is 'Galerkin', which uses a supremizer enrichment
        strategy for the velocity space in order to guarantee inf-sup stability.
        'ls-ls' solves the least-squares problem directly via least_square = True in apply_inverse. 
        'ls-normal' solves the least-squares problem via the normal equation.
    product_u
        Inner product |Operator| w.r.t. which `RB_u` is orthonormalized.
        If `None`, the Euclidean inner product is used.
    product_p 
        Inner product |Operator| w.r.t. which `RB_p` is orthonormalized. 
        If `None`, the Euclidean inner product is used.
    check_orthonormality
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    check_tol
        See :class:`~pymor.reductors.basic.ProjectionBasedReductor`.
    """

    def __init__(self, fom, RB_u=None, RB_p=None, projection_method='Galerkin', product_u=None, product_p=None, check_orthonormality=None, check_tol=None):
        assert isinstance(fom, StationaryModel)
        assert fom.name == "Stokes-TH"
        RB_u = fom.solution_space.subspaces[0].empty() if RB_u is None else RB_u
        RB_p = fom.solution_space.subspaces[1].empty() if RB_p is None else RB_p
        assert RB_u in fom.solution_space.subspaces[0]
        assert RB_p in fom.solution_space.subspaces[1]

        self.projection_method = projection_method

        super().__init__(fom, {'RB_u': RB_u, 'RB_p': RB_p}, {'RB_u': product_u, 'RB_p': product_p},
                        check_orthonormality=check_orthonormality, check_tol=check_tol)

    def project_operators(self):
        fom = self.fom
        RB_u = self.bases['RB_u']
        RB_p = self.bases['RB_p']

        product_u = self.products['RB_u']
        product_p = self.products['RB_p']

        if self.projection_method == 'Galerkin':
            supremizer_space = self.supremizer_enrichment()
            RB_u_enriched = cat_arrays([RB_u, supremizer_space])
            RB_u_enriched = gram_schmidt(RB_u_enriched, product=product_u, offset=len(RB_u))
            self.bases['RB_u_enriched'] = RB_u_enriched

            trial_space = BlockVectorSpace((RB_u_enriched.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u_enriched, RB_p))

            proj_op = project(fom.operator, range_basis=V_block, source_basis=V_block)
            proj_rhs = project(fom.rhs, range_basis=V_block, source_basis=None)
            
            projected_operators = {
                'operator': proj_op,
                'rhs': proj_rhs,
                'products': {'u': project(fom.products['u'], RB_u, RB_u), 'p': project(fom.products['p'], RB_p, RB_p)}
            }

        elif self.projection_method == 'ls-normal':
            trial_space = BlockVectorSpace((RB_u.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u, RB_p))
            
            if product_u and product_p:
                mixed_product = BlockDiagonalOperator(blocks=[product_u, product_p])
                X_h_inv = InverseOperator(mixed_product)
            else:
                X_h_inv = None

            proj_op = project(AdjointOperator(fom.operator, range_product=X_h_inv) @ fom.operator, range_basis=V_block, source_basis=V_block)
            proj_rhs = project(AdjointOperator(fom.operator, range_product=X_h_inv) @ fom.rhs, range_basis=V_block, source_basis=None)

            projected_operators = {
                'operator':         proj_op,
                'rhs':              proj_rhs,
                'products':         {'u': project(fom.products['u'], RB_u, RB_u), 'p': project(fom.products['p'], RB_p, RB_p)}
            }

        elif self.projection_method == 'ls-ls':
            trial_space = BlockVectorSpace((RB_u.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u, RB_p))

            from pymor.algorithms.simplify import expand
            # moves the parameter out of the block and creates a LincombOperator
            expanded_op = expand(fom.operator)

            if product_u and product_p:
                mixed_product = BlockDiagonalOperator(blocks=[product_u, product_p])
                X_h_inv = InverseOperator(mixed_product)
            else:
                X_h_inv = None

            from pymor.algorithms.image import estimate_image
            test_space = estimate_image(operators=[expanded_op], domain=V_block, orthonormalize=True, product=X_h_inv)

            proj_op = project(fom.operator, range_basis=test_space, source_basis=V_block)
            proj_rhs = project(fom.rhs, range_basis=test_space, source_basis=None)

            projected_operators = {
                'operator':         proj_op,
                'rhs':              proj_rhs,
                'products':         {'u': project(fom.products['u'], RB_u, RB_u), 'p': project(fom.products['p'], RB_p, RB_p)}
            }

        else:
            raise NotImplementedError

        return projected_operators

    def supremizer_enrichment(self):
        fom = self.fom
        RB_p = self.bases['RB_p']
        product_u = self.products['RB_u']
        
        block_pu = fom.operator.blocks[1,0]
        supremizer_space = fom.solution_space.subspaces[0].empty()
    
        supremizer_rhs = block_pu.apply_adjoint(RB_p)
        if product_u: 
            supremizer_vector = product_u.apply_inverse(supremizer_rhs)
        else: 
            supremizer_vector = supremizer_rhs

        supremizer_space.append(supremizer_vector)

        return supremizer_space

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim_u = dims['RB_u']
        dim_p = dims['RB_p']
        product_u = self.products['RB_u']
        product_p = self.products['RB_p']

        if self.projection_method == 'Galerkin':
            dim_u = dims['RB_u_enriched']
            projected_operators = {
                'operator': project_to_subbasis(rom.operator, dim_u + dim_p, dim_u + dim_p),
                'rhs': project_to_subbasis(rom.rhs, dim_u + dim_p, None), 
                'products': {'RB_u': project_to_subbasis(product_u, dim_u, dim_u), 'RB_p': project_to_subbasis(product_p, dim_p, dim_p)}
            }

        elif self.projection_method == 'ls-normal' or self.projection_method == 'ls-ls':
            projected_operators = {
            'operator':          project_to_subbasis(rom.operator, dim_u + dim_p, dim_u + dim_p),
            'rhs':               project_to_subbasis(rom.rhs, dim_u + dim_p, None),
            'products':          {'RB_u': project_to_subbasis(product_u, dim_u, dim_u), 'RB_p': project_to_subbasis(product_p, dim_p, dim_p)}
        }
                
        else: 
            raise NotImplementedError

        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        if self.projection_method == 'ls-ls':
            return StationaryModel(error_estimator=error_estimator, least_squares=True, **projected_operators)
        else: 
            return  StationaryModel(error_estimator=error_estimator, **projected_operators)