import numpy as np

from pymor.algorithms.projection import project, ProjectRules
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.models.symplectic import BaseQuadraticHamiltonianModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray
from pymor.vectorarrays.block import BlockVectorSpace


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
        

        print("checking biorthogonality of POD_PH", np.linalg.norm((np.identity(len(V_r)) - (W_r.inner(V_r)))))

        projected_operators = {
                    'H_op':              project(fom.H_op, V_r, V_r),
                    'h':                 project(fom.h, V_r, None),
                    'initial_data':      NumpyMatrixOperator(W_r.inner(fom.initial_data.as_vector())),
                    'J':                 project(CanonicalSymplecticFormOperator(fom.H_op.source), W_r, W_r)
                }
        
        
        
        rom = BaseQuadraticHamiltonianModel(
            T=fom.T, 
            time_stepper=fom.time_stepper, 
            num_values=fom.num_values,
            **projected_operators
            )
        
        return rom
    
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
        