# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.projection import project, project_to_subbasis
from pymor.models.iosys import PHLTIModel
from pymor.reductors.basic import ProjectionBasedReductor


class PHLTIPGReductor(ProjectionBasedReductor):
    """Petrov-Galerkin projection of an |PHLTIModel|.

    Parameters
    ----------
    fom
        The full order |PHLTIModel| to reduce.
    V
        The basis of the trail space.
    E_biorthonormal
        If `True`, no `E` matrix will be assembled for the reduced |Model|.
        Set to `True` if `W` and `V` are biorthonormal w.r.t. `fom.E`.
    """

    def __init__(self, fom, V, E_biorthonormal=False):
        assert isinstance(fom, PHLTIModel)
        super().__init__(fom, {'W': V, 'V': V})
        self.E_biorthonormal = E_biorthonormal

    def project_operators(self):
        fom = self.fom
        W = self.bases['W']
        V = self.bases['V']
        projected_operators = {'E': None if self.E_biorthonormal else project(fom.E, W, V),
                               'J': project(fom.J, W, V),
                               'R': project(fom.R, W, V),
                               'G': project(fom.G, W, None),
                               'P': project(fom.P, W, None),
                               'S': fom.S,
                               'N': fom.N}
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        if dims['W'] != dims['V']:
            raise ValueError
        rom = self._last_rom
        dim = dims['V']
        projected_operators = {'E': None if self.E_biorthonormal else project_to_subbasis(rom.E, dim, dim),
                               'J': project_to_subbasis(rom.J, dim, dim),
                               'R': project_to_subbasis(rom.R, dim, dim),
                               'G': project_to_subbasis(rom.B, dim, None),
                               'P': project_to_subbasis(rom.P, dim, None),
                               'S': rom.S,
                               'N': rom.N}
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        return PHLTIModel(error_estimator=error_estimator, **projected_operators)

    def extend_basis(self, **kwargs):
        raise NotImplementedError

    def reconstruct(self, u, basis='V'):
        return super().reconstruct(u, basis)
