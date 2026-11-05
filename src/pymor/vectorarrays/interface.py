# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from typing_extensions import Self

from pymor.core.base import BasicObject, ImmutableObject, abstractmethod
from pymor.core.defaults import defaults
from pymor.tools.random import get_rng

if TYPE_CHECKING:
    from pymor.operators.interface import Operator

Ind:     TypeAlias = list[int] | slice | None
IndLike: TypeAlias = Ind | int | np.ndarray


class VectorArray(BasicObject):
    """Interface for vector arrays.

    A vector array should be thought of as a list of (possibly high-dimensional) vectors.
    While the vectors themselves will be inaccessible in general (e.g. because they are
    managed by an external PDE solver code), operations on the vectors like addition can
    be performed via this interface.

    It is assumed that the number of vectors is small enough such that scalar data
    associated to each vector can be handled on the Python side. As such, methods like
    :meth:`~VectorArray.norm` or :meth:`~VectorArray.gramian` will
    always return |NumPy arrays|.

    An implementation of the `VectorArray` via |NumPy arrays| is given by
    |NumpyVectorArray|.  In general, it is the implementors decision how memory is
    allocated internally (e.g.  continuous block of memory vs. list of pointers to the
    individual vectors.) Thus, no general assumptions can be made on the costs of operations
    like appending to or removing vectors from the array. As a hint for 'continuous block
    of memory' implementations, :meth:`~VectorSpace.zeros` provides a `reserve`
    keyword argument which allows to specify to what size the array is assumed to grow.

    As with |NumPy array|, |VectorArrays| can be indexed with numbers, slices and
    lists or one-dimensional |NumPy arrays|. Indexing will always return a new
    |VectorArray| which acts as a view into the original data. Thus, if the indexed
    array is modified via :meth:`~VectorArray.scal` or :meth:`~VectorArray.axpy`,
    the vectors in the original array will be changed. Indices may be negative, in
    which case the vector is selected by counting from the end of the array. Moreover
    indices can be repeated, in which case the corresponding vector is selected several
    times. The resulting view will be immutable, however.

    .. note::
        It is disallowed to append vectors to a |VectorArray| view or to remove
        vectors from it. Removing vectors from an array with existing views
        will lead to undefined behavior of these views. As such, it is generally
        advisable to make a :meth:`~VectorArray.copy` of a view for long
        term storage. Since :meth:`~VectorArray.copy` has copy-on-write
        semantics, this will usually cause little overhead.

    Attributes
    ----------
    dim
        The dimension of the vectors in the array.
    is_view
        `True` if the array is a view obtained by indexing another array.
    space
        The |VectorSpace| the array belongs to.
    base
        In case the array is a view, the |VectorArray| this array refers to. `None`
        otherwise.
    ind
        In case the array is a view, the indices with which the :attr:`~VectorArray.base`
        array is indexed with.
    real
        The real part of the vectors in the array.
    imag
        The imaginary part of the vectors in the array.
    """

    # override NumPy binary operations and ufuncs
    __array_priority__ = 100.0
    __array_ufunc__ = None

    impl_type: 'type[VectorArrayImpl]'
    is_view = False
    _impl: 'VectorArrayImpl | None' = None
    base: 'VectorArray | None' = None
    ind: Ind = None

    @property
    def impl(self) -> 'VectorArrayImpl':
        return self.base._impl if self.is_view else self._impl  # type: ignore

    def __init__(self, space: 'VectorSpace', impl: 'VectorArrayImpl | None', base: 'VectorArray | None' = None,
                 ind: Ind = None, _len: int | None = None):
        assert impl is None or isinstance(impl, self.impl_type)
        self.space = space
        if base is None:
            assert impl is not None
            assert ind is None
            self._impl = impl
            self._refcount = [1]
            self._len = len(impl)
        else:
            assert base is not None
            assert _len is not None
            self.is_view = True
            self.base = base
            self.ind = ind
            self._len = _len

    def zeros(self, count: int = 1, reserve: int = 0) -> 'VectorArray':
        """Create a |VectorArray| of null vectors of the same |VectorSpace|.

        This is a shorthand for `self.space.zeros(count, reserve)`.

        Parameters
        ----------
        count
            The number of vectors.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        A |VectorArray| containing `count` vectors with each DOF set to zero.
        """
        return self.space.zeros(count, reserve=reserve)

    def ones(self, count: int = 1, reserve: int = 0) -> 'VectorArray':
        """Create a |VectorArray| of vectors of the same |VectorSpace| with all DOFs set to one.

        This is a shorthand for `self.space.full(1., count, reserve)`.

        Parameters
        ----------
        count
            The number of vectors.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        A |VectorArray| containing `count` vectors with each DOF set to one.
        """
        return self.space.full(1., count, reserve)

    def full(self, value: float | complex, count: int = 1, reserve: int = 0) -> 'VectorArray':
        """Create a |VectorArray| of vectors with all DOFs set to the same value.

        This is a shorthand for `self.space.full(value, count, reserve)`.

        Parameters
        ----------
        value
            The value each DOF should be set to.
        count
            The number of vectors.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        A |VectorArray| containing `count` vectors with each DOF set to `value`.
        """
        return self.space.full(value, count, reserve=reserve)

    def random(self, count:int = 1, distribution: str = 'uniform', reserve: int = 0, **kwargs) -> 'VectorArray':
        """Create a |VectorArray| of vectors with random entries.

        This is a shorthand for
        `self.space.random(count, distribution, **kwargs)`.

        Supported random distributions::

            'uniform': Uniform distribution in half-open interval
                       [`low`, `high`).
            'normal':  Normal (Gaussian) distribution with mean
                       `loc` and standard deviation `scale`.

        Note that not all random distributions are necessarily implemented
        by all |VectorSpace| implementations.

        Parameters
        ----------
        count
            The number of vectors.
        distribution
            Random distribution to use (`'uniform'`, `'normal'`).
        low
            Lower bound for `'uniform'` distribution (defaults to `0`).
        high
            Upper bound for `'uniform'` distribution (defaults to `1`).
        loc
            Mean for `'normal'` distribution (defaults to `0`).
        scale
            Standard deviation for `'normal'` distribution (defaults to `1`).
        reserve
            Hint for the backend to which length the array will grow.
        """
        return self.space.random(count, distribution, reserve, **kwargs)

    def empty(self, reserve: int = 0) -> 'VectorArray':
        """Create an empty |VectorArray| of the same |VectorSpace|.

        This is a shorthand for `self.space.zeros(0, reserve)`.

        Parameters
        ----------
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        An empty |VectorArray|.
        """
        return self.space.zeros(0, reserve=reserve)

    @property
    def dim(self) -> int:
        return self.space.dim

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, ind: IndLike) -> 'VectorArray':
        """Return a |VectorArray| view onto a subset of the vectors in the array."""
        l = self._len

        # normalize ind s.t. the length of the view does not change when
        # the array is appended to
        if isinstance(ind, int):
            if 0 <= ind < l:
                ind = slice(ind, ind+1)
            elif ind >= l or ind < -l:
                raise IndexError('VectorArray index out of range')
            else:
                ind = l+ind
                ind = slice(ind, ind+1)
            view_len = 1
        elif type(ind) is slice:
            start, stop, step = ind.indices(l)
            if start == stop:
                ind = slice(0, 0, 1)
                view_len = 0
            else:
                assert start >= 0
                assert stop >= 0 or (step < 0 and stop >= -1)
                ind = slice(start, None if stop == -1 else stop, step)
                view_len = len(range(start, stop, step))
        else:
            assert isinstance(ind, list | np.ndarray)
            assert all(-l <= i < l for i in ind)
            ind = [i if 0 <= i else l+i for i in ind]
            view_len = len(ind)

        if self.is_view:
            assert self.base is not None
            ind = self._sub_index(self.base._len, self.ind, ind)
            return type(self)(self.space, None, self.base, ind, view_len)
        else:
            return type(self)(self.space, None, self, ind, view_len)

    def __delitem__(self, ind: IndLike) -> None:
        """Remove vectors from the array."""
        if self.is_view:
            raise ValueError('Cannot delete items from VectorArray view')
        assert self.check_ind(ind)
        self._copy_impl_if_multiple_refs()
        self.impl.delete(ind)
        self._len = len(self.impl)

    def to_numpy(self, ensure_copy: bool = False) -> np.ndarray:
        """Return (self.dim, len(self)) NumPy array with the data stored in the array.

        Parameters
        ----------
        ensure_copy
            If `False`, modifying the returned |NumPy array| might alter the original
            |VectorArray|. If `True` always a copy of the array data is made.
        """
        return self.impl.to_numpy(ensure_copy, self.ind)

    def append(self, other: 'VectorArray', remove_from_other=False) -> None:
        """Append vectors to the array.

        Parameters
        ----------
        other
            A |VectorArray| containing the vectors to be appended.
        remove_from_other
            If `True`, the appended vectors are removed from `other`.
            For list-like implementations this can be used to prevent
            unnecessary copies of the involved vectors.
        """
        assert other in self.space
        if self.is_view:
            raise ValueError('Cannot append to VectorArray view')
        if remove_from_other and self.impl is other.impl:
            raise ValueError('Cannot append VectorArray to itself with remove_from_other=True')
        self._copy_impl_if_multiple_refs()
        if remove_from_other:
            other._copy_impl_if_multiple_refs()
        self.impl.append(other.impl, remove_from_other, other.ind)
        self._len += other._len
        if remove_from_other:
            if other.is_view:
                other.base._len = len(other.base._impl)  # type: ignore
            else:
                other._len = 0

    def copy(self, deep: bool = False) -> 'VectorArray':
        """Returns a copy of the array.

        All |VectorArray| implementations in pyMOR have copy-on-write semantics:
        if not specified otherwise by setting `deep` to `True`, the returned
        copy will hold a handle to the same array data as the original array,
        and a deep copy of the data will only be performed when one of the arrays
        is modified.

        Note that for |NumpyVectorArray|, a deep copy is always performed when only
        some vectors in the array are copied.

        Parameters
        ----------
        deep
            Ensure that an actual copy of the array data is made (see above).

        Returns
        -------
        A copy of the |VectorArray|.
        """
        if self.is_view or deep:
            return type(self)(self.space, self.impl.copy(deep, self.ind))
        else:
            C = type(self)(self.space, self.impl)
            C._refcount = self._refcount
            self._refcount[0] += 1
            return C

    def __deepcopy__(self, memo) -> 'VectorArray':
        del memo  # memo is unused
        return self.copy(deep=True)

    def scal(self, alpha: float | complex) -> None:
        """BLAS SCAL operation (in-place scalar multiplication).

        This method calculates ::

            self = alpha*self

        If `alpha` is a scalar, each vector is multiplied by this scalar. Otherwise, `alpha`
        has to be a one-dimensional |NumPy array| of the same length as `self`
        containing the factors for each vector.

        Parameters
        ----------
        alpha
            The scalar coefficient or one-dimensional |NumPy array| of coefficients
            with which the vectors in `self` are multiplied.
        """
        assert self.base is None or self.base.check_ind_unique(self.ind)
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)
        self._copy_impl_if_multiple_refs()
        self.impl.scal(alpha, self.ind)

    def axpy(self, alpha: float | complex, x: 'VectorArray') -> None:
        """BLAS AXPY operation.

        This method forms the sum ::

            self = alpha*x + self

        If the length of `x` is 1, the same `x` vector is used for all vectors
        in `self`. Otherwise, the lengths of `self`  and `x` have to agree.
        If `alpha` is a scalar, each `x` vector is multiplied with the same factor `alpha`.
        Otherwise, `alpha` has to be a one-dimensional |NumPy array| of the same length as
        `self` containing the coefficients for each `x` vector.

        Parameters
        ----------
        alpha
            The scalar coefficient or one-dimensional |NumPy array| of coefficients with which
            the vectors in `x` are multiplied.
        x
            A |VectorArray| containing the x-summands.
        """
        assert self.base is None or self.base.check_ind_unique(self.ind)
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (len(self),)
        assert x in self.space
        assert len(self) == len(x) or len(x) == 1
        self._copy_impl_if_multiple_refs()
        self.impl.axpy(alpha, x.impl, self.ind, x.ind)

    def inner(self, other: 'VectorArray', product: 'Operator | None' = None) -> np.ndarray:
        """Returns the inner products between |VectorArray| elements.

        If `product` is `None`, the Euclidean inner product between
        the :meth:`dofs` of `self` and `other` are returned, i.e. ::

            U.inner(V)

        is equivalent to::

            U.dofs.T(np.arange(U.dim)) @ V.dofs(np.arange(V.dim))

        (Note, that :meth:`dofs` is only intended to be called for a
        small number of DOF indices.)

        If a `product` |Operator| is specified, this |Operator| is
        used to compute the inner products using
        :meth:`~pymor.operators.interface.Operator.apply2`, i.e.
        `U.inner(V, product)` is equivalent to::

            product.apply2(U, V)

        which in turn is, by default, implemented as::

            U.inner(product.apply(V))

        In the case of complex numbers, this is antilinear in the
        first argument, i.e. in 'self'.
        Complex conjugation is done in the first argument because
        most numerical software in the community handles it this way:
        Numpy, DUNE, FEniCS, Eigen, Matlab and BLAS do complex conjugation
        in the first argument, only PETSc and deal.II do complex
        conjugation in the second argument.

        Parameters
        ----------
        other
            A |VectorArray| containing the second factors.
        product
            If not `None` an |Operator| representing the inner product
            bilinear form.

        Returns
        -------
        A |NumPy array| `result` such that:

            result[i, j] = ( self[i], other[j] ).
        """
        if product is not None:
            return product.apply2(self, other)
        else:
            assert self.space == other.space
            return self.impl.inner(other.impl, self.ind, other.ind)

    def pairwise_inner(self, other: 'VectorArray', product: 'Operator | None' = None) -> np.ndarray:
        """Returns the pairwise inner products between |VectorArray| elements.

        If `product` is `None`, the Euclidean inner product between
        the :meth:`dofs` of `self` and `other` are returned, i.e. ::

            U.pairwise_inner(V)

        is equivalent to::

            np.sum(U.dofs(np.arange(U.dim)) * V.dofs(np.arange(V.dim)), axis=0)

        (Note, that :meth:`dofs` is only intended to be called for a
        small number of DOF indices.)

        If a `product` |Operator| is specified, this |Operator| is
        used to compute the inner products using
        :meth:`~pymor.operators.interface.Operator.pairwise_apply2`, i.e.
        `U.inner(V, product)` is equivalent to::

            product.pairwise_apply2(U, V)

        which in turn is, by default, implemented as::

            U.pairwise_inner(product.apply(V))

        In the case of complex numbers, this is antilinear in the
        first argument, i.e. in 'self'.
        Complex conjugation is done in the first argument because
        most numerical software in the community handles it this way:
        Numpy, DUNE, FEniCS, Eigen, Matlab and BLAS do complex conjugation
        in the first argument, only PETSc and deal.II do complex
        conjugation in the second argument.


        Parameters
        ----------
        other
            A |VectorArray| containing the second factors.
        product
            If not `None` an |Operator| representing the inner product
            bilinear form.

        Returns
        -------
        A |NumPy array| `result` such that:

            result[i] = ( self[i], other[i] ).

        """
        if product is not None:
            return product.pairwise_apply2(self, other)
        else:
            assert self.space == other.space
            assert len(self) == len(other)
            return self.impl.pairwise_inner(other.impl, self.ind, other.ind)

    def lincomb(self, coefficients: np.ndarray) -> 'VectorArray':
        """Returns linear combinations of the vectors contained in the array.

        Parameters
        ----------
        coefficients
            A |NumPy array| of dimension 1 or 2 containing the linear
            coefficients. `coefficients.shape[0]` has to agree with
            `len(self)`.

        Returns
        -------
        A |VectorArray| `result` such that:

            result[i] = ∑ self[j] * coefficients[j,i]

        in case `coefficients` is of dimension 2, otherwise
        `len(result) == 1` and

            result[0] = ∑ self[j] * coefficients[j].
        """
        assert 1 <= coefficients.ndim <= 2
        if coefficients.ndim == 1:
            coefficients = coefficients[..., np.newaxis]
        assert coefficients.shape[0] == len(self)
        return type(self)(self.space, self.impl.lincomb(coefficients, self.ind))

    def norm(self, product: 'Operator | None' = None, tol = None, raise_complex = None) -> np.ndarray:
        """Norm with respect to a given inner product.

        If `product` is `None`, the Euclidean norms of the :meth:`dofs`
        of the array are returned, i.e. ::

            U.norm()

        is equivalent to::

            np.sqrt(U.pairwise_inner(U))

        If a `product` |Operator| is specified, this |Operator| is
        used to compute the norms via::

            np.sqrt(product.pairwise_apply2(U, U))

        Parameters
        ----------
        product
            If not `None`, the inner product |Operator| used to compute the
            norms.
        tol
            If `raise_complex` is `True`, a :exc:`ValueError` exception
            is raised if there are complex norm squares with an imaginary part
            of absolute value larger than `tol`.
        raise_complex
            See `tol`.

        Returns
        -------
        A one-dimensional |NumPy array| of the norms of the vectors in the array.
        """
        if product is not None:
            norm_squared = self.norm2(product=product, tol=tol, raise_complex=raise_complex)
            return np.sqrt(norm_squared.real)
        else:
            norm = self.impl.norm(self.ind)
            assert np.all(np.isrealobj(norm))
            return norm

    @defaults('tol', 'raise_complex')
    def norm2(self, product: 'Operator | None' = None, tol = 1e-10, raise_complex = True) -> np.ndarray:
        """Squared norm with respect to a given inner product.

        If `product` is `None`, the Euclidean norms of the :meth:`dofs`
        of the array are returned, i.e. ::

            U.norm()

        is equivalent to::

            U.pairwise_inner(U)

        If a `product` |Operator| is specified, this |Operator| is
        used to compute the norms via::

            product.pairwise_apply2(U, U)

        Parameters
        ----------
        product
            If not `None`, the inner product |Operator| used to compute the
            norms.
        tol
            If `raise_complex` is `True`, a :exc:`ValueError` exception
            is raised if there are complex norm squares with an imaginary part
            of absolute value larger than `tol`.
        raise_complex
            See `tol`.

        Returns
        -------
        A one-dimensional |NumPy array| of the squared norms of the vectors in the array.
        """
        if product is not None:
            norm_squared = product.pairwise_apply2(self, self)
            if raise_complex and np.any(np.abs(norm_squared.imag) > tol):
                raise ValueError(f'norm is complex (square = {norm_squared})')
            return norm_squared.real
        else:
            norm_squared = self.impl.norm2(self.ind)
            assert np.all(np.isrealobj(norm_squared))
            return norm_squared

    def sup_norm(self) -> np.ndarray:
        """The l-infinity-norms of the vectors contained in the array.

        Returns
        -------
        A |NumPy array| `result` such that `result[i]` contains the norm
        of `self[i]`.
        """
        if self.dim == 0:
            return np.zeros(len(self))
        else:
            _, max_val = self.amax()
            return max_val

    def dofs(self, dof_indices: list[int] | np.ndarray) -> np.ndarray:
        """Extract DOFs of the vectors contained in the array.

        Parameters
        ----------
        dof_indices
            List or 1D |NumPy array| of indices of the DOFs that are to be returned.

        Returns
        -------
        A |NumPy array| `result` such that `result[i, j]` is the `dof_indices[i]`-th
        DOF of the `j`-th vector of the array.
        """
        assert isinstance(dof_indices, list) and (len(dof_indices) == 0 or min(dof_indices) >= 0) \
            or (isinstance(dof_indices, np.ndarray) and dof_indices.ndim == 1
                and (len(dof_indices) == 0 or np.min(dof_indices) >= 0))
        # NumPy 1.9 is quite permissive when indexing arrays of size 0, so we have to add the
        # following check:
        assert len(self) > 0 \
            or (isinstance(dof_indices, list)
                and (len(dof_indices) == 0 or max(dof_indices) < self.dim)) \
            or (isinstance(dof_indices, np.ndarray) and dof_indices.ndim == 1
                and (len(dof_indices) == 0 or np.max(dof_indices) < self.dim))
        return self.impl.dofs(np.asarray(dof_indices, dtype=np.int64), self.ind)

    def amax(self) -> tuple[np.ndarray, np.ndarray]:
        """The maximum absolute value of the DOFs contained in the array.

        Returns
        -------
        max_ind
            |NumPy array| containing for each vector a DOF index at which the maximum is
            attained.
        max_val
            |NumPy array| containing for each vector the maximum absolute value of its
            DOFs.
        """
        assert self.dim > 0
        return self.impl.amax(self.ind)

    def gramian(self, product: 'Operator | None' = None) -> np.ndarray:
        """Shorthand for `self.inner(self, product)`."""
        if product is not None:
            return product.apply2(self, self)
        else:
            return self.impl.gramian(self.ind)

    @property
    def real(self) -> 'VectorArray':
        impl = self.impl.real(self.ind)
        if impl is self.impl:
            return self.copy()
        else:
            return type(self)(self.space, impl)

    @property
    def imag(self) -> 'VectorArray':
        """Imaginary part."""
        return type(self)(self.space, self.impl.imag(self.ind))

    def conj(self) -> 'VectorArray':
        """Complex conjugation."""
        impl = self.impl.conj(self.ind)
        if impl is self.impl:
            return self.copy()
        else:
            return type(self)(self.space, impl)

    def __add__(self, other: 'float | complex | VectorArray') -> 'VectorArray':
        if isinstance(other, int | float | complex):
            assert other == 0
            return self.copy()
        assert other in self.space
        assert len(self) == len(other) or len(other) == 1
        return type(self)(self.space, self.impl.axpy_copy(1, other.impl, self.ind, other.ind))

    def __iadd__(self, other: 'VectorArray') -> Self:
        self.axpy(1, other)
        return self

    __radd__ = __add__

    def __sub__(self, other: 'VectorArray') -> 'VectorArray':
        assert other in self.space
        assert len(self) == len(other) or len(other) == 1
        return type(self)(self.space, self.impl.axpy_copy(-1, other.impl, self.ind, other.ind))

    def __isub__(self, other: 'VectorArray') -> Self:
        self.axpy(-1, other)
        return self

    def __mul__(self, other: 'float | complex | VectorArray') -> 'VectorArray':
        assert isinstance(other, int | float | complex) \
            or isinstance(other, np.ndarray) and other.shape == (len(self),)
        return type(self)(self.space, self.impl.scal_copy(other, self.ind))

    __rmul__ = __mul__

    def __imul__(self, other: 'float | complex') -> Self:
        self.scal(other)
        return self

    def __neg__(self) -> 'VectorArray':
        return type(self)(self.space, self.impl.scal_copy(-1, self.ind))

    def check_ind(self, ind) -> bool:
        """Check if index is admissible.

        Check if `ind` is an admissible list of indices in the sense
        of the class documentation.
        """
        l = len(self)
        return (type(ind) is slice
                or isinstance(ind, int) and -l <= ind < l
                or isinstance(ind, list | np.ndarray) and all(-l <= i < l for i in ind))

    def check_ind_unique(self, ind) -> bool:
        """Check if index is admissible and unique.

        Check if `ind` is an admissible list of non-repeated indices in
        the sense of the class documentation.
        """
        l = len(self)
        return (type(ind) is slice
                or isinstance(ind, int) and -l <= ind < l
                or isinstance(ind, list | np.ndarray)
                and len({i if i >= 0 else l+i for i in ind if -l <= i < l}) == len(ind))

    def len_ind(self, ind: IndLike) -> int:
        """Return the number of given indices."""
        return self.impl.len_ind(ind)

    def len_ind_unique(self, ind: IndLike) -> int:
        """Return the number of specified unique indices."""
        l = len(self)
        if ind is None:
            return l
        elif type(ind) is slice:
            return len(range(*ind.indices(l)))
        elif isinstance(ind, int):
            return 1
        else:
            return len({i if i >= 0 else l+i for i in ind})

    def _copy_impl_if_multiple_refs(self) -> None:
        array = self if self.base is None else self.base
        if array._refcount[0] == 1:
            return
        array._impl = array.impl.copy(False, None)  # copy the array implementation
        array._refcount[0] -= 1                     # decrease refcount for original array
        array._refcount = [1]                       # create new reference counter

    @staticmethod
    def _sub_index(l, ind, ind_ind) -> Ind:
        if type(ind) is slice:
            ind = range(*ind.indices(l))
            if type(ind_ind) is slice:
                result = ind[ind_ind]
                return slice(result.start, result.stop, result.step)
            elif hasattr(ind_ind, '__len__'):
                return [ind[i] for i in ind_ind]
            else:
                return [ind[ind_ind]]
        else:
            if not hasattr(ind, '__len__'):
                ind = [ind]
            if type(ind_ind) is slice:
                return ind[ind_ind]
            elif hasattr(ind_ind, '__len__'):
                return [ind[i] for i in ind_ind]
            else:
                return [ind[ind_ind]]

    def __del__(self) -> None:
        if not self.is_view:
            self._refcount[0] -= 1


class VectorSpace(ImmutableObject):
    """Class describing a vector space.

    Vector spaces act as factories for |VectorArrays| of vectors
    contained in them. As such, they hold all data necessary to
    create |VectorArrays| of a given type (e.g. the dimension of
    the vectors, or a socket for communication with an external
    PDE solver).

    New |VectorArrays| of null vectors are created via
    :meth:`~VectorSpace.zeros`.  The
    :meth:`~VectorSpace.make_array` method builds a new
    |VectorArray| from given raw data of the underlying linear algebra
    backend (e.g. a |NumPy array| in the case  of |NumpyVectorSpace|).
    Some vector spaces can create new |VectorArrays| from a given
    |NumPy array| via the :meth:`~VectorSpace.from_numpy`
    method.

    Vector spaces can be compared for equality via the `==` and `!=`
    operators. To test if a given |VectorArray| is an element of
    the space, the `in` operator can be used.


    Attributes
    ----------
    id
        None, or a string describing the mathematical identity
        of the vector space (for instance to distinguish different
        components in an equation system).
    dim
        The dimension (number of degrees of freedom) of the
        vectors contained in the space.
    is_scalar
        Equivalent to `isinstance(space, NumpyVectorSpace) and space.dim == 1`.
    """

    dim: int
    is_scalar = False

    @abstractmethod
    def make_array(*args, **kwargs) -> VectorArray:
        """Create a |VectorArray| from raw data.

        This method is used in the implementation of |Operators|
        and |Models| to create new |VectorArrays| from
        raw data of the underlying solver backends. The ownership
        of the data is transferred to the newly created array.

        The exact signature of this method depends on the wrapped
        solver backend.
        """
        ...

    @abstractmethod
    def zeros(self, count: int = 1, reserve: int = 0) -> VectorArray:
        """Create a |VectorArray| of null vectors.

        Parameters
        ----------
        count
            The number of vectors.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        A |VectorArray| containing `count` vectors with each component zero.
        """
        ...

    def ones(self, count: int = 1, reserve: int = 0) -> VectorArray:
        """Create a |VectorArray| of vectors with all DOFs set to one.

        This is a shorthand for `self.full(1., count, reserve)`.

        Parameters
        ----------
        count
            The number of vectors.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        A |VectorArray| containing `count` vectors with each DOF set to one.
        """
        return self.full(1., count, reserve)

    def full(self, value: float | complex, count: int = 1, reserve: int = 0) -> VectorArray:
        """Create a |VectorArray| of vectors with all DOFs set to the same value.

        Parameters
        ----------
        value
            The value each DOF should be set to.
        count
            The number of vectors.
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        A |VectorArray| containing `count` vectors with each DOF set to `value`.
        """
        del reserve  # reserve unused
        return self.from_numpy(np.full((self.dim, count), value))

    def random(self, count=1, distribution='uniform', reserve=0, **kwargs) -> VectorArray:
        """Create a |VectorArray| of vectors with random entries.

        Supported random distributions::

            'uniform': Uniform distribution in half-open interval
                       [`low`, `high`).
            'normal':  Normal (Gaussian) distribution with mean
                       `loc` and standard deviation `scale`.

        Note that not all random distributions are necessarily implemented
        by all |VectorSpace| implementations.

        Parameters
        ----------
        count
            The number of vectors.
        distribution
            Random distribution to use (`'uniform'`, `'normal'`).
        low
            Lower bound for `'uniform'` distribution (defaults to `0`).
        high
            Upper bound for `'uniform'` distribution (defaults to `1`).
        loc
            Mean for `'normal'` distribution (defaults to `0`).
        scale
            Standard deviation for `'normal'` distribution (defaults to `1`).
        reserve
            Hint for the backend to which length the array will grow.
        """
        del reserve # reserve unused
        values = _create_random_values((self.dim, count), distribution, **kwargs)
        return self.from_numpy(values)

    def empty(self, reserve=0) -> VectorArray:
        """Create an empty |VectorArray|.

        This is a shorthand for `self.zeros(0, reserve)`.

        Parameters
        ----------
        reserve
            Hint for the backend to which length the array will grow.

        Returns
        -------
        An empty |VectorArray|.
        """
        return self.zeros(0, reserve=reserve)

    def from_numpy(self, data, ensure_copy=False) -> VectorArray:
        """Create a |VectorArray| from a |NumPy array|.

        Note that this method will not be supported by all vector
        space implementations.

        Parameters
        ----------
        data
            |NumPy| array of shape `(dim, len)` where `len` is the
            number of vectors and `dim` their dimension.
        ensure_copy
            If `False`, modifying the returned |VectorArray| might alter the original
            |NumPy array|. If `True` always a copy of the array data is made.

        Returns
        -------
        A |VectorArray| with `data` as data.
        """
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        return other is self

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __contains__(self, other) -> bool:
        return self == getattr(other, 'space', None)

    @abstractmethod
    def __hash__(self): ...


def _create_random_values(shape, distribution, order='F', **kwargs):
    if distribution not in ('uniform', 'normal'):
        raise NotImplementedError

    rng = get_rng()

    if distribution == 'uniform':
        if not kwargs.keys() <= {'low', 'high'}:
            raise ValueError
        low = kwargs.get('low', 0.)
        high = kwargs.get('high', 1.)
        if high <= low:
            raise ValueError
        return rng.uniform(low, high, shape).ravel().reshape(shape, order=order)
    elif distribution == 'normal':
        if not kwargs.keys() <= {'loc', 'scale'}:
            raise ValueError
        loc = kwargs.get('loc', 0.)
        scale = kwargs.get('scale', 1.)
        return rng.normal(loc, scale, shape).ravel().reshape(shape, order=order)
    else:
        assert False


class VectorArrayImpl(BasicObject):
    """Implementation of a |VectorArray|.

    The |VectorArray| base class defers all calls to interface methods to an
    internal `impl` object of this type. Indexing, error checking or non-standard
    inner products are handled by |VectorArray|. Possible indices are passed
    to the methods of :class:`~pymor.vectorarrays.interface.VectorArrayImpl` as
    `ind`, `oind` or `xind` parameters. These can either be `None`, in case the
    array has not been indexed, a `slice` object, or a Python list of
    non-negative numbers.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def to_numpy(self, ensure_copy: bool, ind: Ind) -> np.ndarray: ...

    @abstractmethod
    def append(self, other: 'VectorArrayImpl', remove_from_other: bool, oind: Ind) -> None: ...

    @abstractmethod
    def delete(self, ind: Ind) -> None: ...

    @abstractmethod
    def copy(self, deep: bool, ind: Ind) -> 'VectorArrayImpl': ...

    @abstractmethod
    def scal(self, alpha: float | complex, ind: Ind) -> None: ...

    def scal_copy(self, alpha: float | complex, ind: Ind) -> 'VectorArrayImpl':
        result = self.copy(True, ind)
        result.scal(alpha, None)
        return result

    @abstractmethod
    def axpy(self, alpha: float | complex, x: 'VectorArrayImpl', ind: Ind, xind: Ind) -> None: ...

    def axpy_copy(self, alpha: float | complex, x: 'VectorArrayImpl', ind: Ind, xind: Ind) -> 'VectorArrayImpl':
        result = self.copy(True, ind)
        result.axpy(alpha, x, None, xind)
        return result

    @abstractmethod
    def inner(self, other: 'VectorArrayImpl', ind: Ind, oind: Ind) -> np.ndarray: ...

    @abstractmethod
    def pairwise_inner(self, other: 'VectorArrayImpl', ind: Ind, oind: Ind) -> np.ndarray: ...

    def gramian(self, ind: Ind) -> np.ndarray:
        return self.inner(self, ind, ind)

    @abstractmethod
    def lincomb(self, coefficients: np.ndarray, ind: Ind) -> 'VectorArrayImpl': ...

    def norm(self, ind: Ind) -> np.ndarray:
        return np.sqrt(self.norm2(ind))

    @abstractmethod
    def norm2(self, ind: Ind) -> np.ndarray: ...

    @abstractmethod
    def dofs(self, dof_indices: np.ndarray, ind: Ind) -> np.ndarray: ...

    @abstractmethod
    def amax(self, ind: Ind) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def real(self, ind: Ind) -> 'VectorArrayImpl': ...

    @abstractmethod
    def imag(self, ind: Ind) -> 'VectorArrayImpl': ...

    @abstractmethod
    def conj(self, ind: Ind) -> 'VectorArrayImpl': ...

    def len_ind(self, ind: IndLike) -> int:
        if ind is None:
            return len(self)
        if type(ind) is slice:
            return len(range(*ind.indices(len(self))))
        if isinstance(ind, int):
            return 1
        return len(ind)
