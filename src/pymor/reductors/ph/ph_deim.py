# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.projection import project
from pymor.models.nonlinear_ph import NonlinearPHModel
from pymor.operators.constructions import IdentityOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator, ProjectedEmpiricalInterpolatedOperator
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace


class PHdeimReductor(ProjectionBasedReductor):
    """Structure-preserving DEIM reductor for nonlinear port-Hamiltonian systems.

    This reductor constructs a reduced-order model for a |NonlinearPHModel| using a
    structure-preserving Petrov--Galerkin projection combined with the port-Hamiltonian
    discrete empirical interpolation method (pH-DEIM), see Algorithm 5 in :cite:`CBG16`.

    The interpolation DOFs and the collateral basis can be generated using
    the algorithms provided in the :mod:`pymor.algorithms.ei` module.


    Parameters
    ----------
    fom
        The full order |NonlinearPHModel| to reduce.
    V
        The basis of the trail space.
    U
        The collateral basis as a |VectorArray| contained in
        ``fom.dh.range``. Its vectors are used to construct
        the |EmpiricalInterpolatedOperator| approximation of the
        nonlinear operator ``fom.dh``.
    Z
        List or 1D |NumPy array| of the interpolation DOFs.
    QTE_orthonormal
        If `True`, no `E` matrix will be assembled for the reduced |Model|.
        Set to `True` if `V` is orthonormal w.r.t. `fom.Q.H @ fom.E`.
    """

    def __init__(self, fom, V, U, Z, QTE_orthonormal):
        assert isinstance(fom, NonlinearPHModel)

        W = fom.Q.apply(V)

        super().__init__(fom, {'V': V, 'W': W, 'U': U})

        self.QTE_orthonormal = QTE_orthonormal
        self.U = U
        self.Z = Z

    def project_operators(self):
        fom = self.fom
        V = self.bases['V']
        U = self.bases['U']
        W = self.bases['W']
        Z = self.Z

        ei_dh = EmpiricalInterpolatedOperator(fom.dh, Z, U, triangular=False)

        ET_U = U.dofs(Z)
        UT_V = U.inner(V)
        C = np.linalg.solve(ET_U.T, UT_V)

        # Calling project(ei_dh, ...) would require instantiating a high-dimensional array for
        # the projected source basis P^T V, where P = E (E^T U)^{-1} E^T is the DEIM projector
        # and E is the selection matrix for the interpolation DOFs Z. Instead, we construct
        # ProjectedEmpiricalInterpolatedOperator directly: since P^T V = E C is row-sparse
        # (nonzero only at the DEIM DOFs Z), its restriction to source_dofs reduces to selecting
        # the rows of C whose DOF indices appear in source_dofs, via np.intersect1d, staying
        # entirely in reduced dimensions.
        source_dofs = ei_dh.source_dofs
        source_basis_dofs = np.zeros((len(source_dofs), len(V)))
        _, z_idx, s_idx = np.intersect1d(Z, source_dofs, return_indices=True)
        source_basis_dofs[s_idx] = C[z_idx]

        pei_dh = ProjectedEmpiricalInterpolatedOperator(ei_dh.restricted_operator,
                                                        ei_dh.interpolation_matrix,
                                                        NumpyVectorSpace.make_array(source_basis_dofs),
                                                        NumpyVectorSpace.make_array(V.inner(U)),
                                                        triangular=False)

        projected_operators = {'E': None if self.QTE_orthonormal else project(fom.E, W, V),
                               'J': project(fom.J, W, W),
                               'R': project(fom.R, W, W),
                               'G': project(fom.G, W, None),
                               'dh': pei_dh,
                               'Q': IdentityOperator(pei_dh.source),
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
