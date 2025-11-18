import numpy as np

from pymor.core.defaults import defaults
from pymor.models.interface import Model
from pymor.models.generic import GenericModel
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class OrthogonalProjectionReductor(ProjectionBasedReductor):
    """Reductor performing an orthogonal projection onto a reduced basis.

    _Note_ that the resulting model does not bring any computational benefits.

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

        # Calculate μ-independent parts once per ROM - approx 100x faster than recomputing in loop
        B_sub = self.basis[:dim]
        G = B_sub.gramian()

        def project_onto_basis(U):
            # See https://docs.pymor.org/2024-1-2/tutorial_basis_generation.html#a-trivial-reduced-basis
            R = B_sub.inner(U)
            return np.linalg.solve(G, R).T

        rom_space = NumpyVectorSpace(dim)
        rom = GenericModel(
            parameters=self.fom.parameters,
            computers={
                "solution": (
                    rom_space,
                    lambda mu: rom_space.from_numpy(
                        project_onto_basis(self.fom.solve(mu))
                    ),
                ),
            },
            name=f"BestApproximationReduced{self.fom.name}",
        )
        rom.disable_logging()
        return rom
