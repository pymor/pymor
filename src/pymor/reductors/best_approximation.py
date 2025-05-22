import numpy as np

from pymor.core.defaults import defaults
from pymor.models.black_box import BlackBoxModel, NumpyBlackBoxModel
from pymor.models.interface import Model
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


class BestApproximationReductor(ProjectionBasedReductor):
    """Generic reductor using best-approximation onto a reduced basis.

    _Note_ that the resulting model does does not bring any computational benefits.

    We achieve restriction to sub-basis by using the projected operators as a dimension tag.
    """

    @defaults('check_orthonormality', 'check_tol')
    def __init__(
        self,
        fom: Model,
        basis=None, # dict or vectorarray
        check_orthonormality=True,
        check_tol=1e-3,
    ):
        if isinstance(basis, (list, tuple)):
            assert isinstance(fom.solution_space, BlockVectorSpace)
            assert len(basis) == len(fom.solution_space.subspaces)
            assert all(b in s for b, s in zip(basis, fom.solution_space.subspaces))
            basis = {f'RB_{i}': b for i, b in enumerate(basis)}
        else:
            basis = basis or fom.solution_space.empty()
            assert basis in fom.solution_space
            basis = {'RB': basis}
        super().__init__(
            fom,
            basis,
            check_orthonormality=check_orthonormality,
            check_tol=check_tol,
        )
        self.__auto_init(locals())

    def project_operators(self):
        return {key: len(basis) for key, basis in self.basis.items()}

    def project_operators_to_subbasis(self, dims):
        return dims

    def reconstruct(self, u):
        if len(self.bases) == 1:
            super().reconstruct(u)
        else:
            assert isinstance(self.fom.solution_space, BlockVectorSpace)
            assert isinstance(u.space, BlockVectorSpace)
            Us_blocks = [None for i in range(len(self.basis))]
            for i in range(len(self.basis)):
                basis = self.basis[f'RB_{i}']
                u_block = u.blocks[i]
                dim = u_block.dim
                assert dim <= len(basis)
                Us_blocks[i] = basis[:dim].lincomb(u_block.to_numpy())
            return self.fom.solution_space.make_array(Us_blocks)

    def build_rom(self, projected_operators, error_estimator):
        if len(self.bases) == 1:
            # TODO: this should conceptually work for a BlockVectorSpace as well,
            # but I did not test project_onto_basis and subsequent solve
            assert not isinstance(self.fom.solution_space, BlockVectorSpace)
            dim = projected_operators['RB']
            assert dim <= len(self.basis['RB'])

            def project_onto_basis(U):
                # See https://docs.pymor.org/2024-1-2/tutorial_basis_generation.html#a-trivial-reduced-basis
                G = self.basis[:dim].gramian()
                R = self.basis[:dim].inner(U)
                return np.linalg.solve(G, R)

            rom = NumpyBlackBoxModel(
                dim,
                self.fom.parameters,
                lambda mu: project_onto_basis(self.fom.solve(mu)),
            )
            rom.disable_logging()
            return rom

        else:
            assert isinstance(self.fom.solution_space, BlockVectorSpace)
            dims = projected_operators
            assert isinstance(dims, dict)
            assert all(key in self.basis for key in dims)
            max_dim = max(dims.values())
            assert all(dims[key] <= max_dim for key in self.basis)
            dims = [min(dims[f'RB_{i}'], max_dim) for i in range(len(self.basis))]
            # dims = {key: min(dim, max_dim) for key, dim in self.dims.items()}

            blocked_RB_space = BlockVectorSpace(NumpyVectorSpace(d) for d in dims)

            def project_onto_basis(blocked_U):
                # TODO: replace zeros by something uninitialised?
                projected_U = np.zeros((np.sum(dims), len(blocked_U)))
                for i in range(len(self.basis)):
                    U = blocked_U.blocks[i]
                    basis = self.basis[f'RB_{i}']
                    dim = dims[i]
                    # See https://docs.pymor.org/2024-1-2/tutorial_basis_generation.html#a-trivial-reduced-basis
                    G = basis[:dim].gramian()
                    R = basis[:dim].inner(U)
                    u = np.linalg.solve(G, R)
                    assert u.shape == (dim, len(blocked_U))
                    projected_U[int(np.sum(dims[:i])):int(np.sum(dims[:i]) + dim), :] = u[:, :]
                return blocked_RB_space.from_numpy(projected_U)

            rom = BlackBoxModel(
                blocked_RB_space,
                self.fom.parameters,
                lambda mu: project_onto_basis(self.fom.solve(mu)),
            )
            rom.disable_logging()
            return rom
