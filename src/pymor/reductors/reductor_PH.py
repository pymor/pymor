import numpy as np

from pymor.algorithms.projection import project, ProjectRules
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.models.symplectic import BaseQuadraticHamiltonianModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.operators.constructions import ConcatenationOperator, IdentityOperator, InverseOperator
from pymor.models.basic import InstationaryModel


class PHReductor():
    def __init__(self, fom, V_r, W_r):
        self.fom = fom
        self.V_r = V_r
        self.W_r = W_r

    def reduce(self):
        fom = self.fom
        V_r = self.V_r
        W_r = self.W_r
        # numpy_W_r_transposed = W_r.to_numpy().transpose()
        # half_dim = W_r.dim//2
        # space = BlockVectorSpace([NumpyVectorSpace(half_dim), NumpyVectorSpace(half_dim)])
        # numpy_W_r_transposed = space.from_numpy(np.array([numpy_W_r_transposed[:half_dim, :], numpy_W_r_transposed[half_dim:, :]]))
        

        # print("checking biorthogonality of POD_PH", np.linalg.norm((np.identity(len(V_r)) - (W_r.inner(V_r)))))
        H_op_proj = project(fom.H_op, V_r, V_r)
        n = H_op_proj.source.dim // 2
        J = NumpyMatrixOperator(
            np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]]))
        
        projection_matrix = W_r.inner(V_r)
        print("projection matrix - Identity norm", np.linalg.norm(np.identity(len(V_r)) - projection_matrix))
        projection_op = NumpyMatrixOperator(projection_matrix)
        inverse_projection_op = InverseOperator(projection_op, 'inverse_projection_op')
        pid = project(fom.initial_data, range_basis=W_r, source_basis=None, product=None)
        projected_initial_data = ConcatenationOperator([inverse_projection_op, pid])
        print("difference", np.linalg.norm(((projected_initial_data - project(fom.initial_data, W_r, None)).as_vector().to_numpy())))

        print("mass function fom", fom.mass)
        

        projected_operators = {
            'mass':              project(fom.mass, V_r, V_r),
            'operator':          project(fom.operator, V_r, V_r),
            'rhs':               project(fom.rhs, V_r, None),
            'initial_data':      projected_initial_data,
            'products':          {k: project(v, V_r, V_r) for k, v in fom.products.items()},
            'output_functional': project(fom.output_functional, None, V_r)
        }
        
        rom = InstationaryModel(T=fom.T, time_stepper=fom.time_stepper, num_values=fom.num_values,
                                 error_estimator=None, **projected_operators)
        
        return rom
    
class ReducedPHModel(InstationaryModel):
    def __init__(self, T, time_stepper, num_values, projected_operators, error_estimator):

        # generate J as NumpyMatrixOperator

        return 
    
class check_PODReductor():
    def __init__(self, fom, V_r):
        self.fom = fom
        self.V_r = V_r

    def reduce(self):
        fom = self.fom
        V_r = self.V_r
        print(len(V_r), V_r.dim)
        print("checking orthogonality of check_POD", np.linalg.norm((np.identity(len(V_r)) - (V_r.inner(V_r)))))
        H_op_proj = project(fom.H_op, V_r, V_r)
        # assert H_op_proj.source.dim % 2 == 0
        n = H_op_proj.source.dim // 2
        J_1 = NumpyMatrixOperator(np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]]))
        J_2 = project(CanonicalSymplecticFormOperator(fom.H_op.source), V_r, V_r)
        projected_operators = {
                    'H_op':              H_op_proj,
                    'h':                 project(fom.h, V_r, None),
                    'initial_data':      project(fom.initial_data, V_r, None),
                    'J':                 J_2
                }
  
        
        rom = BaseQuadraticHamiltonianModel(
            T=fom.T, 
            time_stepper=fom.time_stepper, 
            num_values=fom.num_values,
            **projected_operators
            )
        
        return rom
        