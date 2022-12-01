# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.core.base import BasicObject, abstractmethod
from pymor.core.defaults import defaults
from pymor.core.exceptions import ExtensionError, AccuracyError
from pymor.models.basic import StationaryModel, InstationaryModel
from pymor.models.iosys import LTIModel, SecondOrderModel, LinearDelayModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import ConcatenationOperator, IdentityOperator, InverseOperator


class ProjectionBasedReductor(BasicObject):
    """Generic projection based reductor.

    Parameters
    ----------
    fom
        The full order |Model| to reduce.
    bases
        A dict of |VectorArrays| of basis vectors.
    products
        A dict of inner product |Operators| w.r.t. which the corresponding bases are
        orthonormalized. A value of `None` corresponds to orthonormalization of the
        basis w.r.t. the Euclidean inner product.
    check_orthonormality
        If `True`, check if bases which have a corresponding entry in the `products`
        dict are orthonormal w.r.t. the given inner product. After each
        :meth:`basis extension <extend_basis>`, orthonormality is checked again.
    check_tol
        If `check_orthonormality` is `True`, the numerical tolerance with which the checks
        are performed.
    """

    @defaults('check_orthonormality', 'check_tol')
    def __init__(self, fom, bases, products={}, check_orthonormality=True, check_tol=1e-3):
        assert products.keys() <= bases.keys()
        bases = dict(bases)
        products = dict(products)
        self.__auto_init(locals())
        self._last_rom = None

        if check_orthonormality:
            for basis in bases:
                self._check_orthonormality(basis)

    def reduce(self, dims=None):
        if dims is None:
            dims = {k: len(v) for k, v in self.bases.items()}
        if isinstance(dims, Number):
            dims = {k: dims for k in self.bases}
        if set(dims.keys()) != set(self.bases.keys()):
            raise ValueError(f'Must specify dimensions for {set(self.bases.keys())}')
        for k, d in dims.items():
            if d < 0:
                raise ValueError(f'Reduced state dimension must be larger than zero {k}')
            if d > len(self.bases[k]):
                raise ValueError(f'Specified reduced state dimension larger than reduced basis {k}')

        if self._last_rom is None or any(dims[b] > self._last_rom_dims[b] for b in dims):
            self._last_rom = self._reduce()
            self._last_rom_dims = {k: len(v) for k, v in self.bases.items()}

        if dims == self._last_rom_dims:
            return self._last_rom
        else:
            return self._reduce_to_subbasis(dims)

    def _reduce(self):
        with self.logger.block('Operator projection ...'):
            projected_operators = self.project_operators()

        # ensure that no logging output is generated for error_estimator assembly in case there is
        # no error estimator to assemble
        if self.assemble_error_estimator.__func__ is not ProjectionBasedReductor.assemble_error_estimator:
            with self.logger.block('Assembling error estimator ...'):
                error_estimator = self.assemble_error_estimator()
        else:
            error_estimator = None

        with self.logger.block('Building ROM ...'):
            rom = self.build_rom(projected_operators, error_estimator)
            rom = rom.with_(name=f'{self.fom.name}_reduced')
            rom.disable_logging()

        return rom

    def _reduce_to_subbasis(self, dims):
        projected_operators = self.project_operators_to_subbasis(dims)
        error_estimator = self.assemble_error_estimator_for_subbasis(dims)
        rom = self.build_rom(projected_operators, error_estimator)
        rom = rom.with_(name=f'{self.fom.name}_reduced')
        rom.disable_logging()
        return rom

    @abstractmethod
    def project_operators(self):
        pass

    def assemble_error_estimator(self):
        return None

    @abstractmethod
    def build_rom(self, projected_operators, error_estimator):
        pass

    def project_operators_to_subbasis(self, dims):
        raise NotImplementedError

    def assemble_error_estimator_for_subbasis(self, dims):
        return None

    def reconstruct(self, u, basis='RB'):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.bases[basis][:u.dim].lincomb(u.to_numpy())

    def extend_basis(self, U, basis='RB', method='gram_schmidt', pod_modes=1, pod_orthonormalize=True, copy_U=True):
        basis_length = len(self.bases[basis])

        extend_basis(U, self.bases[basis], self.products.get(basis), method=method, pod_modes=pod_modes,
                     pod_orthonormalize=pod_orthonormalize,
                     copy_U=copy_U)

        self._check_orthonormality(basis, basis_length)

    def _check_orthonormality(self, basis, offset=0):
        if not self.check_orthonormality or basis not in self.products:
            return

        U = self.bases[basis]
        product = self.products.get(basis, None)
        error_matrix = U[offset:].inner(U, product)
        error_matrix[:len(U) - offset, offset:] -= np.eye(len(U) - offset)
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= self.check_tol:
                raise AccuracyError(f"result not orthogonal (max err={err})")


class StationaryRBReductor(ProjectionBasedReductor):
    """Galerkin projection of a |StationaryModel|.

    Parameters
    ----------
    fom
        The full order |Model| to reduce.
    RB
        The basis of the reduced space onto which to project. If `None` an empty basis is used.
    product
        Inner product |Operator| w.r.t. which `RB` is orthonormalized. If `None`, the Euclidean
        inner product is used.
    check_orthonormality
        See :class:`ProjectionBasedReductor`.
    check_tol
        See :class:`ProjectionBasedReductor`.
    """

    def __init__(self, fom, RB=None, product=None, check_orthonormality=None, check_tol=None):
        assert isinstance(fom, StationaryModel)
        RB = fom.solution_space.empty() if RB is None else RB
        assert RB in fom.solution_space
        super().__init__(fom, {'RB': RB}, {'RB': product},
                         check_orthonormality=check_orthonormality, check_tol=check_tol)

    def project_operators(self):
        fom = self.fom
        RB = self.bases['RB']
        projected_operators = {
            'operator':          project(fom.operator, RB, RB),
            'rhs':               project(fom.rhs, RB, None),
            'products':          {k: project(v, RB, RB) for k, v in fom.products.items()},
            'output_functional': project(fom.output_functional, None, RB)
        }
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim = dims['RB']
        projected_operators = {
            'operator':          project_to_subbasis(rom.operator, dim, dim),
            'rhs':               project_to_subbasis(rom.rhs, dim, None),
            'products':          {k: project_to_subbasis(v, dim, dim) for k, v in rom.products.items()},
            'output_functional': project_to_subbasis(rom.output_functional, None, dim)
        }
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        return StationaryModel(error_estimator=error_estimator, **projected_operators)


class InstationaryRBReductor(ProjectionBasedReductor):
    """Galerkin projection of an |InstationaryModel|.

    Parameters
    ----------
    fom
        The full order |Model| to reduce.
    RB
        The basis of the reduced space onto which to project. If `None` an empty basis is used.
    product
        Inner product |Operator| w.r.t. which `RB` is orthonormalized. If `None`, the
        the Euclidean inner product is used.
    initial_data_product
        Inner product |Operator| w.r.t. which the `initial_data` of `fom` is orthogonally projected.
        If `None`, the Euclidean inner product is used.
    product_is_mass
        If `True`, no mass matrix for the reduced |Model| is assembled.  Set to `True` if `RB` is
        orthonormal w.r.t. the `mass` matrix of `fom`.
    check_orthonormality
        See :class:`ProjectionBasedReductor`.
    check_tol
        See :class:`ProjectionBasedReductor`.
    """

    def __init__(self, fom, RB=None, product=None, initial_data_product=None, product_is_mass=False,
                 check_orthonormality=None, check_tol=None):
        assert isinstance(fom, InstationaryModel)
        RB = fom.solution_space.empty() if RB is None else RB
        assert RB in fom.solution_space
        super().__init__(fom, {'RB': RB}, {'RB': product},
                         check_orthonormality=check_orthonormality, check_tol=check_tol)
        self.initial_data_product = initial_data_product or product
        self.product_is_mass = product_is_mass

    def project_operators(self):
        fom = self.fom
        RB = self.bases['RB']
        product = self.products['RB']

        if self.initial_data_product != product:
            # TODO there should be functionality for this somewhere else
            projection_matrix = RB.gramian(self.initial_data_product)
            projection_op = NumpyMatrixOperator(projection_matrix)
            inverse_projection_op = InverseOperator(projection_op, 'inverse_projection_op')
            pid = project(fom.initial_data, range_basis=RB, source_basis=None, product=self.initial_data_product)
            projected_initial_data = ConcatenationOperator([inverse_projection_op, pid])
        else:
            projected_initial_data = project(fom.initial_data, range_basis=RB, source_basis=None,
                                             product=product)

        projected_operators = {
            'mass':              (None if (isinstance(fom.mass, IdentityOperator) and product is None
                                           or self.product_is_mass) else
                                  project(fom.mass, RB, RB)),
            'operator':          project(fom.operator, RB, RB),
            'rhs':               project(fom.rhs, RB, None),
            'initial_data':      projected_initial_data,
            'products':          {k: project(v, RB, RB) for k, v in fom.products.items()},
            'output_functional': project(fom.output_functional, None, RB)
        }

        return projected_operators

    def project_operators_to_subbasis(self, dims):
        rom = self._last_rom
        dim = dims['RB']
        product = self.products['RB']

        if self.initial_data_product != product:
            # TODO there should be functionality for this somewhere else
            pop = project_to_subbasis(rom.initial_data.operators[1], dim_range=dim, dim_source=None)
            inverse_projection_op = InverseOperator(
                project_to_subbasis(rom.initial_data.operators[0].operator, dim_range=dim, dim_source=dim),
                name='inverse_projection_op'
            )
            projected_initial_data = ConcatenationOperator([inverse_projection_op, pop])
        else:
            projected_initial_data = project_to_subbasis(rom.initial_data, dim_range=dim, dim_source=None)

        projected_operators = {
            'mass':              project_to_subbasis(rom.mass, dim, dim),
            'operator':          project_to_subbasis(rom.operator, dim, dim),
            'rhs':               project_to_subbasis(rom.rhs, dim, None),
            'initial_data':      projected_initial_data,
            'products':          {k: project_to_subbasis(v, dim, dim) for k, v in rom.products.items()},
            'output_functional': project_to_subbasis(rom.output_functional, None, dim)
        }
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        fom = self.fom
        return InstationaryModel(T=fom.T, time_stepper=fom.time_stepper, num_values=fom.num_values,
                                 error_estimator=error_estimator, **projected_operators)


class LTIPGReductor(ProjectionBasedReductor):
    """Petrov-Galerkin projection of an |LTIModel|.

    Parameters
    ----------
    fom
        The full order |Model| to reduce.
    W
        The basis of the test space.
    V
        The basis of the ansatz space.
    E_biorthonormal
        If `True`, no `E` matrix will be assembled for the reduced |Model|.
        Set to `True` if `W` and `V` are biorthonormal w.r.t. `fom.E`.
    """

    def __init__(self, fom, W, V, E_biorthonormal=False):
        assert isinstance(fom, LTIModel)
        super().__init__(fom, {'W': W, 'V': V})
        self.E_biorthonormal = E_biorthonormal

    def project_operators(self):
        fom = self.fom
        W = self.bases['W']
        V = self.bases['V']
        projected_operators = {'A': project(fom.A, W, V),
                               'B': project(fom.B, W, None),
                               'C': project(fom.C, None, V),
                               'D': fom.D,
                               'E': None if self.E_biorthonormal else project(fom.E, W, V)}
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        if dims['W'] != dims['V']:
            raise ValueError
        rom = self._last_rom
        dim = dims['V']
        projected_operators = {'A': project_to_subbasis(rom.A, dim, dim),
                               'B': project_to_subbasis(rom.B, dim, None),
                               'C': project_to_subbasis(rom.C, None, dim),
                               'D': rom.D,
                               'E': None if self.E_biorthonormal else project_to_subbasis(rom.E, dim, dim)}
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        return LTIModel(error_estimator=error_estimator, sampling_time=self.fom.sampling_time, **projected_operators)

    def extend_basis(self, **kwargs):
        raise NotImplementedError

    def reconstruct(self, u, basis='V'):
        return super().reconstruct(u, basis)


class SOLTIPGReductor(ProjectionBasedReductor):
    """Petrov-Galerkin projection of an |SecondOrderModel|.

    Parameters
    ----------
    fom
        The full order |Model| to reduce.
    W
        The basis of the test space.
    V
        The basis of the ansatz space.
    E_biorthonormal
        If `True`, no `E` matrix will be assembled for the reduced |Model|.
        Set to `True` if `W` and `V` are biorthonormal w.r.t. `fom.E`.
    """

    def __init__(self, fom, W, V, M_biorthonormal=False):
        assert isinstance(fom, SecondOrderModel)
        super().__init__(fom, {'W': W, 'V': V})
        self.M_biorthonormal = M_biorthonormal

    def project_operators(self):
        fom = self.fom
        W = self.bases['W']
        V = self.bases['V']
        projected_operators = {'M':  None if self.M_biorthonormal else project(fom.M, W, V),
                               'E':  project(fom.E, W, V),
                               'K':  project(fom.K, W, V),
                               'B':  project(fom.B, W, None),
                               'Cp': project(fom.Cp, None, V),
                               'Cv': project(fom.Cv, None, V),
                               'D':  fom.D}
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        if dims['W'] != dims['V']:
            raise ValueError
        rom = self._last_rom
        dim = dims['V']
        projected_operators = {'M':  None if self.M_biorthonormal else project_to_subbasis(rom.M, dim, dim),
                               'E':  project_to_subbasis(rom.E, dim, dim),
                               'K':  project_to_subbasis(rom.K, dim, dim),
                               'B':  project_to_subbasis(rom.B, dim, None),
                               'Cp': project_to_subbasis(rom.C, None, dim),
                               'Cv': project_to_subbasis(rom.C, None, dim),
                               'D':  rom.D}
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        return SecondOrderModel(error_estimator=error_estimator, **projected_operators)

    def extend_basis(self, **kwargs):
        raise NotImplementedError

    def reconstruct(self, u, basis='V'):
        return super().reconstruct(u, basis)


class DelayLTIPGReductor(ProjectionBasedReductor):
    """Petrov-Galerkin projection of an |LinearDelayModel|.

    Parameters
    ----------
    fom
        The full order |Model| to reduce.
    W
        The basis of the test space.
    V
        The basis of the ansatz space.
    E_biorthonormal
        If `True`, no `E` matrix will be assembled for the reduced |Model|.
        Set to `True` if `W` and `V` are biorthonormal w.r.t. `fom.E`.
    """

    def __init__(self, fom, W, V, E_biorthonormal=False):
        assert isinstance(fom, LinearDelayModel)
        super().__init__(fom, {'W': W, 'V': V})
        self.E_biorthonormal = E_biorthonormal

    def project_operators(self):
        fom = self.fom
        W = self.bases['W']
        V = self.bases['V']
        projected_operators = {'A': project(fom.A, W, V),
                               'Ad': tuple(project(op, W, V) for op in fom.Ad),
                               'B': project(fom.B, W, None),
                               'C': project(fom.C, None, V),
                               'D': fom.D,
                               'E': None if self.E_biorthonormal else project(fom.E, W, V)}
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        if dims['W'] != dims['V']:
            raise ValueError
        rom = self._last_rom
        dim = dims['V']
        projected_operators = {'A': project_to_subbasis(rom.A, dim, dim),
                               'Ad': tuple(project_to_subbasis(op, dim, dim) for op in rom.Ad),
                               'B': project_to_subbasis(rom.B, dim, None),
                               'C': project_to_subbasis(rom.C, None, dim),
                               'D': rom.D,
                               'E': None if self.E_biorthonormal else project_to_subbasis(rom.E, dim, dim)}
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        return LinearDelayModel(tau=self.fom.tau, error_estimator=error_estimator, **projected_operators)

    def extend_basis(self, **kwargs):
        raise NotImplementedError

    def reconstruct(self, u, basis='V'):
        return super().reconstruct(u, basis)


def extend_basis(U, basis, product=None, method='gram_schmidt', pod_modes=1, pod_orthonormalize=True, copy_U=True):
    assert method in ('trivial', 'gram_schmidt', 'pod')

    basis_length = len(basis)

    if method == 'trivial':
        remove = set()
        for i in range(len(U)):
            if np.any(almost_equal(U[i], basis)):
                remove.add(i)
        basis.append(U[[i for i in range(len(U)) if i not in remove]],
                     remove_from_other=(not copy_U))
    elif method == 'gram_schmidt':
        basis.append(U, remove_from_other=(not copy_U))
        gram_schmidt(basis, offset=basis_length, product=product, copy=False, check=False)
    elif method == 'pod':
        U_proj_err = U - basis.lincomb(U.inner(basis, product))

        basis.append(pod(U_proj_err, modes=pod_modes, product=product, orth_tol=np.inf)[0])

        if pod_orthonormalize:
            gram_schmidt(basis, offset=basis_length, product=product, copy=False, check=False)

    if len(basis) <= basis_length:
        raise ExtensionError
