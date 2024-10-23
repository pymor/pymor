import numpy as np

from pymor.core.defaults import defaults
from pymor.models.interface import Model
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.interface import VectorArray

from models.black_box import BlackBoxModel


class BestApproximationReductor(ProjectionBasedReductor):
    """Generic reductor using best-approximation onto a reduced basis.

    _Note_ that the resulting model does does not bring any computational benefits.

    We achieve restriction to sub-basis by using the projected operators as a dimension tag.
    """

    @defaults("check_orthonormality", "check_tol")
    def __init__(
        self,
        fom: Model,
        basis: VectorArray = None,
        check_orthonormality=True,
        check_tol=1e-3,
    ):
        basis = basis or fom.solution_space.empty()
        assert basis in fom.solution_space
        super().__init__(
            fom,
            {"RB": basis},
            check_orthonormality=check_orthonormality,
            check_tol=check_tol,
        )
        self.__auto_init(locals())

    def project_operators(self):
        return len(self.basis)

    def project_operators_to_subbasis(self, dims):
        dim = dims["RB"]
        assert dim <= len(self.basis)
        return dim

    def build_rom(self, projected_operators, error_estimator):
        dim = projected_operators
        assert dim <= len(self.basis)

        def project_onto_basis(U):
            # See https://docs.pymor.org/2024-1-2/tutorial_basis_generation.html#a-trivial-reduced-basis
            G = self.basis[:dim].gramian()
            R = self.basis[:dim].inner(U)
            return np.linalg.solve(G, R).T

        rom = BlackBoxModel(
            dim,
            self.fom.parameters,
            lambda mu: project_onto_basis(self.fom.solve(mu)),
        )
        rom.disable_logging()
        return rom
