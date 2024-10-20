import numpy as np

from pymor.core.base import BasicObject
from pymor.models.interface import Model
from pymor.vectorarrays.interface import VectorArray

from models.black_box import BlackBoxModel


class BestApproximationReductor(BasicObject):
    """Generic reductor using best-approximation onto a reduced basis.

    _Note_ that the resulting model does does not bring any computational benefits.
    """

    def __init__(self, fom: Model, basis: VectorArray = None):
        basis = basis or fom.solution_space.empty()
        assert basis in fom.solution_space
        self.__auto_init(locals())

    def reduce(self):
        def project_onto_basis(U):
            # See https://docs.pymor.org/2024-1-2/tutorial_basis_generation.html#a-trivial-reduced-basis
            G = self.basis.gramian()
            R = self.basis.inner(U)
            return np.linalg.solve(G, R).T

        return BlackBoxModel(
            len(self.basis),
            self.fom.parameters,
            lambda mu: project_onto_basis(self.fom.solve(mu)),
        )

    def extend_basis(
        self,
        U: VectorArray,
        method="gram_schmidt",
        pod_modes=1,
        pod_orthonormalize=True,
        copy_U=True,
    ):
        from pymor.reductors.basic import extend_basis as _extend_basis

        assert U in self.fom.solution_space
        _extend_basis(
            U,
            self.basis,
            product=None,
            method=method,
            pod_modes=pod_modes,
            pod_orthonormalize=pod_orthonormalize,
            copy_U=copy_U,
        )

    def reconstruct(self, u: VectorArray):
        return self.basis.lincomb(u.to_numpy())
