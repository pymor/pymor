import numpy as np

from pymor.algorithms.projection import project, ProjectRules
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.models.symplectic import BaseQuadraticHamiltonianModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.operators.constructions import ConcatenationOperator, IdentityOperator, InverseOperator
from pymor.models.basic import InstationaryModel
from pymor.algorithms.simplify import contract, expand
from pymor.core.base import BasicObject
from numbers import Number





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
                    'J':                 J_1
                }
  
        
        rom = BaseQuadraticHamiltonianModel(
            T=fom.T, 
            time_stepper=fom.time_stepper, 
            num_values=fom.num_values,
            **projected_operators
            )
        
        return rom
    

class MyQuadraticHamiltonianRBReductor(BasicObject):
    """Symplectic Galerkin projection of a |QuadraticHamiltonianModel|.

    Parameters
    ----------
    fom
        The full order |QuadraticHamiltonianModel| to reduce.
    RB
        A |SymplecticBasis| prescribing the basis vectors.
    """

    def __init__(self, fom, V_r, W_r):
        # assert isinstance(fom, QuadraticHamiltonianModel)
        # if not isinstance(fom.mass, IdentityOperator):
        #     raise NotImplementedError
        # if fom.products:
        #     raise NotImplementedError
        
        # assert RB.phase_space == fom.solution_space

        self.fom = fom
        self.V_r = V_r
        self.W_r = W_r
        self.bases = {'RB': V_r}

    def extend_basis(self, U, method='svd_like', modes=2, copy_U=False):
        self.RB.extend(U, method=method, modes=modes)

    def reduce(self, dims=None):
        with self.logger.block('Operator projection ...'):
            if isinstance(dims, dict):
                dim = dims.get('RB', None)
            elif isinstance(dims, Number) or dims is None:
                dim = dims
            else:
                raise NotImplementedError

            if dim is None:
                dim = len(self.V_r) * 2
            assert dim % 2 == 0, 'dim has to be even'

            fom = self.fom
            # RB = self.V_r[:dim//2]
            projection_matrix = self.W_r.inner(self.V_r)
            projection_op = NumpyMatrixOperator(projection_matrix)
            inverse_projection_op = InverseOperator(projection_op, 'inverse_projection_op')
            pid = project(fom.initial_data, range_basis=self.W_r, source_basis=None, product=None)
            projected_initial_data = ConcatenationOperator([inverse_projection_op, pid])
            # projected_initial_data = project(fom.initial_data, self.W_r, None)

            projected_H_op = project(fom.H_op, self.V_r, self.V_r)
            if len(self.W_r) == 0:
                n = 0
                numpy_W_r = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
            else:
                n = fom.H_op.source.dim // 2
                numpy_W_r = self.W_r.to_numpy()
            n = fom.H_op.source.dim // 2
            J_inside = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
            print(J_inside.shape)
            # J = NumpyMatrixOperator(
            #     (numpy_W_r.transpose() @ J_inside @ numpy_W_r)
            # )
            J = project(CanonicalSymplecticFormOperator(fom.H_op.source), self.W_r, self.W_r)

            projected_operators = {
                'H_op':              projected_H_op,
                'h':                 project(fom.h, self.V_r, None),
                'initial_data':      projected_initial_data,
                'output_functional': project(fom.output_functional, None, self.V_r) if fom.output_functional else None,
                'J':                 J
            }

        with self.logger.block('Building ROM ...'):
            rom = ReducedQuadraticHamiltonianModel(
                fom.T,
                time_stepper=fom.time_stepper,
                num_values=fom.num_values,
                name=None if fom.name is None else 'reduced_' + fom.name,
                **projected_operators
            )
        return rom

    def reconstruct(self, u):
        return self.RB[:u.dim//2].lincomb(u.to_numpy())


class ReducedQuadraticHamiltonianModel(BaseQuadraticHamiltonianModel):
    """A general reduced quadratic Hamiltonian system.

    In contrast to the |QuadraticHamiltonianModel|, the reduced model does not rely on a
    |BlockVectorSpace| as `phase_space`.
    """

    def __init__(self, T, initial_data, J, H_op, h=None, time_stepper=None, nt=None, num_values=None,
                 output_functional=None, visualizer=None, name=None):

        # generate J as NumpyMatrixOperator
        assert H_op.source.dim % 2 == 0

        super().__init__(T, initial_data, J, H_op, h, time_stepper, nt, num_values,
                         output_functional, visualizer, name)

        self.operator = contract(expand(self.operator))
        