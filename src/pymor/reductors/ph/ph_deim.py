# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.ei import deim
from pymor.algorithms.projection import project
from pymor.models.nonlinear_ph import NonlinearPHModel
from pymor.operators.constructions import IdentityOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import ProjectionBasedReductor


class PHdeimReductor(ProjectionBasedReductor):
    """Structure-preserving DEIM reductor for nonlinear port-Hamiltonian systems.

    This reductor constructs a reduced-order model for a |NonlinearPHModel| using a
    structure-preserving Petrov--Galerkin projection combined with the port-Hamiltonian
    discrete empirical interpolation method (pH-DEIM), see Algorithm 5 in :cite:`CBG16`.


    Parameters
    ----------
    fom
        The full order |NonlinearPHModel| to reduce.
    V
        The basis of the trail space.
    G
        The collateral basis as a |VectorArray| contained in
        ``fom.dh.range``. Its vectors are used to construct
        the DEIM approximation of the nonlinear operator ``fom.dh``.
    QTE_orthonormal
        If `True`, no `E` matrix will be assembled for the reduced |Model|.
        Set to `True` if `V` is orthonormal w.r.t. `fom.Q.H @ fom.E`.
    """

    def __init__(self, fom, V, G, QTE_orthonormal):
        assert isinstance(fom, NonlinearPHModel)

        W = fom.Q.apply(V)

        super().__init__(fom, {'V': V, 'W': W, 'G': G})

        self.QTE_orthonormal = QTE_orthonormal

    def project_operators(self):
        fom = self.fom
        V = self.bases['V']
        G = self.bases['G']
        W = self.bases['W']
        J = project(fom.J, W, W)

        interpolation_dofs, U, _ = deim(G, pod=False)

        emp_grad_h = EmpiricalInterpolatedOperator(fom.dh, interpolation_dofs, U, triangular=False)

        ET_U = U.dofs(interpolation_dofs)
        UT_V = U.inner(V)
        C = np.linalg.solve(ET_U.T, UT_V)

        PT_V_numpy = np.zeros((fom.solution_space.dim, len(V)))
        PT_V_numpy[interpolation_dofs, :] = C
        PT_V = fom.solution_space.from_numpy(PT_V_numpy)

        projected_operators = {'E': None if self.QTE_orthonormal else project(fom.E, W, V),
                               'J': J,
                               'R': project(fom.R, W, W),
                               'G': project(fom.G, W, None),
                               'dh': project(emp_grad_h, V, PT_V),
                               'Q': IdentityOperator(J.source),
                               'P': project(fom.P, W, None),
                               'S': fom.S,
                               'N': fom.N}
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        raise NotImplementedError

    def build_rom(self, projected_operators, error_estimator):
        fom = self.fom

        if fom.initial_data is not None:
            initial_data = project(fom.initial_data, self.bases['W'], None)
        else:
            initial_data = None

        return NonlinearPHModel(T=fom.T, initial_data=initial_data, time_stepper=fom.time_stepper, nt=fom.nt,
                                num_values=fom.num_values, error_estimator=error_estimator, **projected_operators)

    def extend_basis(self, **kwargs):
        raise NotImplementedError

    def reconstruct(self, u, basis='V'):
        return super().reconstruct(u, basis)



class PGNonlinearPHReductor(ProjectionBasedReductor):
    def __init__(self, fom, V, QTE_orthonormal):
        assert isinstance(fom, NonlinearPHModel)
        W = fom.Q.apply(V)

        super().__init__(fom, {'V': V, 'W': W})

        self.QTE_orthonormal = QTE_orthonormal

    def project_operators(self):
        fom = self.fom
        V = self.bases['V']
        W = self.bases['W']
        J = project(fom.J, W, W)

        projected_operators = {'E': None if self.QTE_orthonormal else project(fom.E, W, V),
                               'J': J,
                               'R': project(fom.R, W, W),
                               'G': project(fom.G, W, None),
                               'dh': project(fom.dh, V, V),
                               'Q': IdentityOperator(J.source),
                               'P': project(fom.P, W, None),
                               'S': fom.S,
                               'N': fom.N}
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        raise NotImplementedError

    def build_rom(self, projected_operators, error_estimator):
        fom = self.fom

        if fom.initial_data is not None:
            initial_data = project(fom.initial_data, self.bases['W'], None)
        else:
            initial_data = None

        return NonlinearPHModel(T=fom.T, initial_data=initial_data, time_stepper=fom.time_stepper, nt=fom.nt,
                                num_values=fom.num_values, error_estimator=error_estimator, **projected_operators)

    def extend_basis(self, **kwargs):
        raise NotImplementedError

    def reconstruct(self, u, basis='V'):
        return super().reconstruct(u, basis)
