# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.reductors.basic import ProjectionBasedReductor
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import InverseOperator, ZeroOperator, AdjointOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.operators.block import BlockOperator, BlockDiagonalOperator
from pymor.vectorarrays.constructions import cat_arrays


class StationaryRBStokesReductor(ProjectionBasedReductor): 
    """Galerkin or leasst-sqaures projection of a |StationaryModel| for the stationary stokes equation.

    Parameters
    ----------
    fom
        The Stokes full order |Model| to reduce.
    RB_u
        The basis of the reduced velocity space onto which to project. If `None` an empty basis is used.
    RB_p
        The basis of the redcued pressure space onto which to project. If `None` an empty basis is used
    projection_method 
        'Galerkin', 'ls-ls' or 'ls-normal'. Default is 'Galerkin', which uses a supremizer enrichtment strategy for the velocity space in order to guarantee inf-sup stability. 
        'ls-ls' solves the least-squares problem directly via least_square = True in apply_inverse. 'ls-normal' solves the least-squares problem via the normal equation. 
    product_u
        Inner product |Operator| w.r.t. which `RB_u` is orthonormalized. If `None`, the Euclidean
        inner product is used.
    product_p 
        Inner product |Operator| w.r.t. which `RB_p` is orthonormalized. If `None`, the Euclidean
        inner product is used.
    check_orthonormality
        See :class:`ProjectionBasedReductor`.
    check_tol
        See :class:`ProjectionBasedReductor`.
    """

    def __init__(self, fom, RB_u=None, RB_p=None, projection_method='Galerkin', product_u=None, product_p=None, check_orthonormality=None, check_tol=None):
        assert isinstance(fom, StationaryModel)
        assert fom.name == "Stokes-TH"
        RB_u = fom.solution_space.subspaces[0].empty() if RB_u is None else RB_u
        RB_p = fom.solution_space.subspaces[1].empty() if RB_p is None else RB_p
        assert RB_u in fom.solution_space.subspaces[0]
        assert RB_p in fom.solution_space.subspaces[1]

        if projection_method == 'ls-ls' or projection_method == 'ls-normal':
            assert product_u 
            assert product_p

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
            self.bases['RB_u_enriched'] = RB_u_enriched

            trial_space = BlockVectorSpace((RB_u_enriched.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u_enriched, RB_p))

            proj_op = project(fom.operator, range_basis=V_block, source_basis=V_block)
            proj_rhs = project(fom.rhs, range_basis=V_block, source_basis=None)
            
            projected_operators = {
                'operator': proj_op,
                'rhs': proj_rhs,
                'products': {k: project(v, RB_u, RB_u) if k == 'u' else project(v, RB_p, RB_p) for k, v in fom.products.items()}
            }

        elif self.projection_method == 'ls-normal':
            trial_space = BlockVectorSpace((RB_u.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u, RB_p))
            mixed_product = BlockDiagonalOperator(blocks=[product_u, product_p])
            X_h_inv = InverseOperator(mixed_product)
            
            proj_op = project(AdjointOperator(fom.operator) @ X_h_inv @ fom.operator, range_basis=V_block, source_basis=V_block)
            proj_rhs = project(AdjointOperator(fom.operator) @ X_h_inv @ fom.rhs, range_basis=V_block, source_basis=None)

            projected_operators = {
                'operator':         proj_op,
                'rhs':              proj_rhs,
                'products':         {k: project(v, RB_u, RB_u) if k == 'u' else project(v, RB_p, RB_p) for k, v in fom.products.items()}
            }
        
        elif self.projection_method == 'ls-ls': 
            trial_space = BlockVectorSpace((RB_u.space, RB_p.space))
            V_block = trial_space.make_block_diagonal_array((RB_u, RB_p))
            mixed_product = BlockDiagonalOperator(blocks=[product_u, product_p])
            X_h_inv = InverseOperator(mixed_product)

            from pymor.algorithms.image import estimate_image
            test_space = estimate_image(operators=[X_h_inv @ fom.operator], domain=V_block, orthonormalize=True)

            proj_op = project(fom.operator, range_basis=test_space, source_basis=V_block)
            proj_rhs = project(fom.rhs, range_basis=test_space, source_basis=None)

            #from pymor.operators.numpy import NumpyMatrixOperator
            #from pymor.operators.constructions import VectorArrayOperator
            #proj_op = fom.operator @ BlockDiagonalOperator(blocks=[VectorArrayOperator(RB_u), VectorArrayOperator(RB_p)])
            #proj_rhs = fom.rhs

            projected_operators = {
                'operator':         proj_op,
                'rhs':              proj_rhs,
                'products':         {k: project(v, RB_u, RB_u) if k == 'u' else project(v, RB_p, RB_p) for k, v in fom.products.items()} 
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
        supremizer_vector = product_u.apply_inverse(supremizer_rhs)
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
                'products': {'u': project_to_subbasis(product_u, dim_u, dim_u), 'u': project_to_subbasis(product_p, dim_p, dim_p)}
            }

        elif self.projection_method == 'ls-normal' or self.projection_method == 'ls-ls': 
            projected_operators = {
            'operator':          project_to_subbasis(rom.operator, dim_u + dim_p, dim_u + dim_p),
            'rhs':               project_to_subbasis(rom.rhs, dim_u + dim_p, None),
            'products':          {'u': project_to_subbasis(product_u, dim_u, dim_u), 'u': project_to_subbasis(product_p, dim_p, dim_p)}
        }
                
        else: 
            raise NotImplementedError

        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        if self.projection_method == 'ls-ls':
            return StationaryModel(error_estimator=error_estimator, least_squares=True, **projected_operators)
        else: 
            return  StationaryModel(error_estimator=error_estimator, **projected_operators)