# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_FENICS:
    import dolfin as df
    import numpy as np

    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import OperatorBase
    from pymor.operators.complex import ComplexOperator
    from pymor.vectorarrays.interfaces import _create_random_values
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace

    class FenicsVector(CopyOnWriteVector):
        """Wraps a FEniCS vector to make it usable with ListVectorArray."""

        def __init__(self, impl, imag=None):
            self.impl = impl
            self._imag = imag

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl, imag=instance._imag)

        def _copy_data(self):
            self.impl = self.impl.copy()
            if self._imag is not None:
                self._imag = self._imag.copy()

        def to_numpy(self, ensure_copy=False):
            if ensure_copy:
                impl = self.impl.copy().get_local()
                imag = None if self._imag is None else self._imag.copy().get_local()
            else:
                impl = self.impl.get_local()
                imag = None if self._imag is None else self._imag.get_local()
            if imag is None:
                result = impl
            else:
                result = impl + imag * 1j
            return result

        def _scal(self, alpha):
            if self._imag is None:
                if alpha.imag != 0:
                    self._imag = alpha.imag * self.impl
                self.impl *= alpha.real
            else:
                if alpha.imag == 0:
                    self.impl *= alpha.real
                    self._imag *= alpha.real
                else:
                    self.impl, self._imag = (
                        alpha.real * self.impl - alpha.imag * self._imag,
                        alpha.imag * self.impl + alpha.real * self._imag
                    )

        def _axpy(self, alpha, x):
            if x is self:
                self.scal(1. + alpha)
            else:
                # real part
                self.impl.axpy(alpha.real, x.impl)
                if x._imag is not None:
                    self.impl.axpy(-alpha.imag, x._imag)

                # imaginary part
                if self._imag is None:
                    if alpha.imag != 0 and x._imag is None:
                        self._imag = alpha.imag * x.impl
                    elif alpha.imag == 0 and x._imag is not None:
                        self._imag = alpha.real * x._imag
                    elif alpha.imag != 0 and x._imag is not None:
                        self._imag = alpha.imag * x.impl
                        self._imag.axpy(alpha.real, x._imag)
                else:
                    if alpha.imag != 0:
                        self._imag.axpy(alpha.imag, x.impl)
                    if x._imag is not None:
                        self._imag.axpy(alpha.real, x._imag)

        def dot(self, other):
            result = self.impl.inner(other.impl)
            if self._imag is not None:
                result += self._imag.inner(other.impl) * (-1j)
            elif other._imag is not None:
                result += self.impl.inner(other._imag) * 1j
            elif self._imag is not None and other._imag is not None:
                result += self._imag.inner(other._imag)
            return result

        def l1_norm(self):
            if self._imag is not None:
                raise NotImplementedError
            return self.impl.norm('l1')

        def l2_norm(self):
            result = self.impl.norm('l2')
            if self._imag is not None:
                result = np.linalg.norm([result, self._imag.norm('l2')])
            return result

        def l2_norm2(self):
            result = self.impl.norm('l2') ** 2
            if self._imag is not None:
                result += self._imag.norm('l2') ** 2
            return result

        def sup_norm(self):
            if self._imag is not None:
                raise NotImplementedError
            return self.impl.norm('linf')

        def dofs(self, dof_indices):
            dof_indices = np.array(dof_indices, dtype=np.intc)
            if len(dof_indices) == 0:
                return np.array([], dtype=np.intc)
            assert 0 <= np.min(dof_indices)
            assert np.max(dof_indices) < self.impl.size()
            dofs = self.impl.gather(dof_indices)
            if self._imag is not None:
                dofs += self._imag.gather(dof_indices)
            # in the mpi distributed case, gather returns the values
            # at the *global* dof_indices on each rank
            return dofs

        def amax(self):
            if self._imag is None:
                A = np.abs(self.impl.get_local())
            else:
                A = np.abs(self.impl.get_local() + self._imag.get_local() * 1j)
            # there seems to be no way in the interface to compute amax without making a copy.
            max_ind_on_rank = np.argmax(A)
            max_val_on_rank = A[max_ind_on_rank]
            from pymor.tools import mpi
            if not mpi.parallel:
                return max_ind_on_rank, max_val_on_rank
            else:
                max_global_ind_on_rank = max_ind_on_rank + self.impl.local_range()[0]
                comm = self.impl.mpi_comm()
                comm_size = comm.Get_size()

                max_inds = np.empty(comm_size, dtype='i')
                comm.Allgather(np.array(max_global_ind_on_rank, dtype='i'), max_inds)

                max_vals = np.empty(comm_size, dtype=np.float64)
                comm.Allgather(np.array(max_val_on_rank), max_vals)

                i = np.argmax(max_inds)
                return max_inds[i], max_vals[i]

        def __add__(self, other):
            impl = self.impl + other.impl
            imag = None
            if self._imag is not None and other._imag is None:
                imag = self._imag.copy()
            elif self._imag is None and other._imag is not None:
                imag = other._imag.copy()
            elif self._imag is not None and other._imag is not None:
                imag = self._imag + other._imag
            return FenicsVector(impl, imag=imag)

        def __iadd__(self, other):
            self._copy_data_if_needed()
            self.impl += other.impl
            if self._imag is None and other._imag is not None:
                self._imag = other._imag.copy()
            elif self._imag is not None and other._imag is not None:
                self._imag += other._imag
            return self

        __radd__ = __add__

        def __sub__(self, other):
            impl = self.impl - other.impl
            imag = None
            if self._imag is not None and other._imag is None:
                imag = self._imag.copy()
            elif self._imag is None and other._imag is not None:
                imag = -other._imag.copy()
            elif self._imag is not None and other._imag is not None:
                imag = self._imag - other._imag
            return FenicsVector(impl, imag=imag)

        def __isub__(self, other):
            self._copy_data_if_needed()
            self.impl -= other.impl
            if self._imag is None and other._imag is not None:
                self._imag = -other._imag.copy()
            elif self._imag is not None and other._imag is not None:
                self._imag -= other._imag
            return self

        def __mul__(self, other):
            impl = self.impl.copy()
            imag = None if self._imag is None else self._imag.copy()
            if imag is None:
                if other.imag != 0:
                    imag = other.imag * impl
                impl *= other.real
            else:
                if other.imag == 0:
                    impl *= other.real
                    imag *= other.real
                else:
                    impl, imag = (
                        other.real * impl - other.imag * imag,
                        other.imag * impl + other.real * imag
                    )
            return FenicsVector(impl, imag=imag)

        def __neg__(self):
            return FenicsVector(-self.impl,
                                imag=None if self._imag is None else -self._imag)

        @property
        def real(self):
            return FenicsVector(self.impl.copy())

        @property
        def imag(self):
            if self._imag is None:
                return FenicsVector(self.impl * 0)
            else:
                return FenicsVector(self._imag.copy())

        def conj(self):
            if self._imag is None:
                return self.copy()
            else:
                return FenicsVector(self.impl.copy(), imag=-self._imag)

    class FenicsVectorSpace(ListVectorSpace):

        def __init__(self, V, id='STATE'):
            self.__auto_init(locals())

        @property
        def dim(self):
            return df.Function(self.V).vector().size()

        def __eq__(self, other):
            return type(other) is FenicsVectorSpace and self.V == other.V and self.id == other.id

        # since we implement __eq__, we also need to implement __hash__
        def __hash__(self):
            return id(self.V) + hash(self.id)

        def zero_vector(self):
            impl = df.Function(self.V).vector()
            return FenicsVector(impl)

        def full_vector(self, value):
            impl = df.Function(self.V).vector()
            impl += value
            return FenicsVector(impl)

        def random_vector(self, distribution, random_state, **kwargs):
            impl = df.Function(self.V).vector()
            values = _create_random_values(impl.local_size(), distribution, random_state, **kwargs)
            impl[:] = values
            return FenicsVector(impl)

        def make_vector(self, obj):
            return FenicsVector(obj)

    class FenicsMatrixOperator(OperatorBase):
        """Wraps a FEniCS matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, source_space, range_space, solver_options=None, name=None, imag=None):
            assert matrix.rank() == 2
            assert imag is None or imag.rank() == 2
            self.__auto_init(locals())
            self.source = FenicsVectorSpace(source_space)
            self.range = FenicsVectorSpace(range_space)

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                # real part
                self.matrix.mult(u.impl, r.impl)
                if self.imag is not None and u._imag is not None:
                    vec = self.range.zero_vector().impl
                    self.imag.mult(u._imag, vec)
                    r.impl.axpy(-1, vec)
                # imaginary part
                if self.imag is not None or u._imag is not None:
                    if u._imag is not None:
                        r._imag = self.range.zero_vector().impl
                        self.matrix.mult(u._imag, r._imag)
                    if self.imag is not None:
                        if r._imag is None:
                            r._imag = self.range.zero_vector().impl
                            self.imag.mult(u.impl, r._imag)
                        else:
                            vec = self.range.zero_vector().impl
                            self.imag.mult(u.impl, vec)
                            r._imag.axpy(1, vec)
            return R

        def apply_adjoint(self, V, mu=None):
            assert V in self.range
            R = self.source.zeros(len(V))
            for v, r in zip(V._list, R._list):
                # real part
                self.matrix.transpmult(v.impl, r.impl)
                if self.imag is not None and v._imag is not None:
                    vec = self.range.zero_vector().impl
                    self.imag.transpmult(v._imag, vec)
                    r.impl.axpy(1, vec)
                # imaginary part
                if self.imag is not None or v._imag is not None:
                    if v._imag is not None:
                        r._imag = self.range.zero_vector().impl
                        self.matrix.transpmult(v._imag, r._imag)
                    if self.imag is not None:
                        if r._imag is None:
                            r._imag = self.range.zero_vector().impl
                            self.imag.transpmult(v.impl, r._imag)
                            r._imag.scal(-1)
                        else:
                            vec = self.range.zero_vector().impl
                            self.imag.transpmult(v.impl, vec)
                            r._imag.axpy(-1, vec)
            return R

        def apply_inverse(self, V, mu=None, least_squares=False):
            assert V in self.range
            if least_squares:
                raise NotImplementedError
            if self.imag is None:
                R = self.source.zeros(len(V))
                options = self.solver_options.get('inverse') if self.solver_options else None
                for r, v in zip(R._list, V._list):
                    _apply_inverse(self.matrix, r.impl, v.impl, options)
                    if v._imag is not None:
                        r._imag = self.range.zero_vector().impl
                        _apply_inverse(self.matrix, r._imag, v._imag, options)
            else:
                real = FenicsMatrixOperator(self.matrix, self.source.V, self.range.V)
                imag = FenicsMatrixOperator(self.imag, self.source.V, self.range.V)
                R = ComplexOperator(real, imag).apply_inverse(V)
            return R

        def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
            if not all(isinstance(op, FenicsMatrixOperator) for op in operators):
                return None
            if identity_shift != 0:
                return None
            assert not solver_options

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
                imag = None if operators[0].imag is None else operators[0].imag.copy()
            elif coefficients[0].imag == 0:
                matrix = operators[0].matrix * coefficients[0].real
                imag = None if operators[0].imag is None else operators[0].imag * coefficients[0].real
            else:
                matrix = operators[0].matrix * coefficients[0].real
                imag = operators[0].matrix * coefficients[0].imag
                if operators[0].imag is not None:
                    matrix.axpy(-coefficients[0].imag, operators[0].imag, False)
                    imag.axpy(coefficients[0].real, operators[0].imag, False)
            for op, c in zip(operators[1:], coefficients[1:]):
                matrix.axpy(c.real, op.matrix, False)
                if c.imag != 0 and op.imag is not None:
                    matrix.axpy(-c.imag, op.imag, False)
                if c.imag != 0:
                    if imag is None:
                        imag = op.matrix * c.imag
                    else:
                        imag.axpy(c.imag, op.matrix, False)
                if op.imag is not None:
                    if imag is None:
                        imag = op.imag * c.real
                    else:
                        imag.axpy(c.real, op.imag, False)
                # in general, we cannot assume the same nonzero pattern for
                # all matrices. how to improve this?

            return FenicsMatrixOperator(matrix, self.source.V, self.range.V, name=name, imag=imag)

    @defaults('solver', 'preconditioner')
    def _solver_options(solver='bicgstab', preconditioner='amg'):
        return {'solver': solver, 'preconditioner': preconditioner}

    def _apply_inverse(matrix, r, v, options=None):
        options = options or _solver_options()
        solver = options.get('solver')
        preconditioner = options.get('preconditioner')
        # preconditioner argument may only be specified for iterative solvers:
        options = (solver, preconditioner) if preconditioner else (solver,)
        df.solve(matrix, r, v, *options)

    class FenicsVisualizer(BasicInterface):
        """Visualize a FEniCS grid function.

        Parameters
        ----------
        space
            The `FenicsVectorSpace` for which we want to visualize DOF vectors.
        """

        def __init__(self, space):
            self.space = space

        def visualize(self, U, m, title='', legend=None, filename=None, block=True,
                      separate_colorbars=True):
            """Visualize the provided data.

            Parameters
            ----------
            U
                |VectorArray| of the data to visualize (length must be 1). Alternatively,
                a tuple of |VectorArrays| which will be visualized in separate windows.
                If `filename` is specified, only one |VectorArray| may be provided which,
                however, is allowed to contain multipled vectors that will be interpreted
                as a time series.
            m
                Filled in by :meth:`pymor.models.ModelBase.visualize` (ignored).
            title
                Title of the plot.
            legend
                Description of the data that is plotted. If `U` is a tuple of |VectorArrays|,
                `legend` has to be a tuple of the same length.
            filename
                If specified, write the data to that file. `filename` needs to have an extension
                supported by FEniCS (e.g. `.pvd`).
            separate_colorbars
                If `True`, use separate colorbars for each subplot.
            block
                If `True`, block execution until the plot window is closed.
            """
            if filename:
                assert not isinstance(U, tuple)
                assert U in self.space
                f = df.File(filename)
                function = df.Function(self.space.V)
                if legend:
                    function.rename(legend, legend)
                for u in U._list:
                    function.vector()[:] = u.impl
                    f << function
            else:
                from matplotlib import pyplot as plt

                assert U in self.space and len(U) == 1 \
                    or (isinstance(U, tuple) and all(u in self.space for u in U) and all(len(u) == 1 for u in U))
                if not isinstance(U, tuple):
                    U = (U,)
                if isinstance(legend, str):
                    legend = (legend,)
                assert legend is None or len(legend) == len(U)

                if not separate_colorbars:
                    vmin = np.inf
                    vmax = -np.inf
                    for u in U:
                        vec = u._list[0].impl
                        vmin = min(vmin, vec.min())
                        vmax = max(vmax, vec.max())

                for i, u in enumerate(U):
                    function = df.Function(self.space.V)
                    function.vector()[:] = u._list[0].impl
                    if legend:
                        tit = title + ' -- ' if title else ''
                        tit += legend[i]
                    else:
                        tit = title
                    if separate_colorbars:
                        plt.figure()
                        df.plot(function, title=tit)
                    else:
                        plt.figure()
                        df.plot(function, title=tit,
                                range_min=vmin, range_max=vmax)
                plt.show(block=block)
