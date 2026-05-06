# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.projection import project, project_to_subbasis
from pymor.models.iosys import LTIModel, PHLTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.basic import ProjectionBasedReductor


class PHLTIPGReductor(ProjectionBasedReductor):
    """Petrov-Galerkin projection of an |PHLTIModel|.

    Parameters
    ----------
    fom
        The full order |PHLTIModel| to reduce.
    V
        The basis of the trail space.
    QTE_orthonormal
        If `True`, no `E` matrix will be assembled for the reduced |Model|.
        Set to `True` if `V` is orthonormal w.r.t. `fom.Q.H @ fom.E`.
    pg_projection
        Choice of test space:

            - ``'ph_preserving'`` (default): :math:`W = Q V`. Yields a reduced
            |PHLTIModel| with all pH structural properties preserved.
            - ``'energy_stable'``: :math:`W = (J - R)^{-T} E V`.
            Yields a reduced system that is no longer of pH structure, 
            but preserves the Hamiltonian part of the structure and thus the energy. 
            Requires :math:`J - R` to be invertible.
    """

    _PG_PROJECTIONS = ('ph_preserving', 'energy_stable')

    def __init__(self, fom, V, QTE_orthonormal=False, pg_projection="ph_preserving"):
        assert isinstance(fom, PHLTIModel)
        if pg_projection not in self._PG_PROJECTIONS:
            raise ValueError(f"Unknown projection {pg_projection!r}. " 
                             f"Expected one of {self._PG_PROJECTIONS}.")
        
        if pg_projection == 'ph_preserving':
            W = fom.Q.apply(V)
        else:
            J_minus_R = fom.J - fom.R
            W = J_minus_R.apply_inverse_adjoint(fom.E.apply(V))

        super().__init__(fom, {'W': W, 'V': V})
        self.QTE_orthonormal = QTE_orthonormal
        self.pg_projection = pg_projection

    def project_operators(self):
        fom = self.fom
        W = self.bases['W']
        V = self.bases['V']

        if self.pg_projection is "ph_preserving":
            J = project(fom.J, W, W)
            projected_operators = {'E': None if self.QTE_orthonormal else project(fom.E, W, V),
                                   'J': J,
                                   'R': project(fom.R, W, W),
                                   'Q': IdentityOperator(J.source),
                                   'G': project(fom.G, W, None),
                                   'P': project(fom.P, W, None),
                                   'S': fom.S,
                                   'N': fom.N}
        else: 
            projected_operators = {'E': project(fom.E, W, V), 
                                   'A': None if self.QTE_orthonormal else project(fom.E.H @ fom.Q, V, V), 
                                   'B': project(fom.G - fom.P, W, None),
                                   'C': project((fom.G + fom.P).H @ fom.Q, None, V),
                                   'D': fom.S - fom.N}
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        if dims['W'] != dims['V']:
            raise ValueError
        rom = self._last_rom
        dim = dims['V']
        J = project_to_subbasis(rom.J, dim, dim)
        projected_operators = {'E': None if self.QTE_orthonormal else project_to_subbasis(rom.E, dim, dim),
                               'J': J,
                               'R': project_to_subbasis(rom.R, dim, dim),
                               'Q': IdentityOperator(J.source),
                               'G': project_to_subbasis(rom.G, dim, None),
                               'P': project_to_subbasis(rom.P, dim, None),
                               'S': rom.S,
                               'N': rom.N}
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        if self.pg_projection is "ph_preserving":
            return PHLTIModel(error_estimator=error_estimator, **projected_operators)
        else: 
            return LTIModel(error_estimator=error_estimator, **projected_operators)

    def extend_basis(self, **kwargs):
        raise NotImplementedError

    def reconstruct(self, u, basis='V'):
        return super().reconstruct(u, basis)
