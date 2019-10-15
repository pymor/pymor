# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_FENICS:
    import dolfin as df
    import numpy as np

    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import LinearComplexifiedListVectorArrayOperatorBase
    from pymor.vectorarrays.interfaces import _create_random_values
    from pymor.vectorarrays.list import CopyOnWriteVector, ComplexifiedVector, ComplexifiedListVectorSpace

    class FenicsVector(CopyOnWriteVector):
        """Wraps a FEniCS vector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            self.impl = self.impl.copy()

        def to_numpy(self, ensure_copy=False):
            if ensure_copy:
                return self.impl.copy().get_local()
            return self.impl.get_local()

        def _scal(self, alpha):
            self.impl *= alpha

        def _axpy(self, alpha, x):
            if x is self:
                self.scal(1. + alpha)
            else:
                self.impl.axpy(alpha, x.impl)

        def dot(self, other):
            return self.impl.inner(other.impl)

        def l1_norm(self):
            return self.impl.norm('l1')

        def l2_norm(self):
            return self.impl.norm('l2')

        def l2_norm2(self):
            return self.impl.norm('l2') ** 2

        def sup_norm(self):
            return self.impl.norm('linf')

        def dofs(self, dof_indices):
            dof_indices = np.array(dof_indices, dtype=np.intc)
            if len(dof_indices) == 0:
                return np.array([], dtype=np.intc)
            assert 0 <= np.min(dof_indices)
            assert np.max(dof_indices) < self.impl.size()
            dofs = self.impl.gather(dof_indices)
            # in the mpi distributed case, gather returns the values
            # at the *global* dof_indices on each rank
            return dofs

        def amax(self):
            raise NotImplementedError  # is implemented for complexified vector

        def __add__(self, other):
            return FenicsVector(self.impl + other.impl)

        def __iadd__(self, other):
            self._copy_data_if_needed()
            self.impl += other.impl
            return self

        __radd__ = __add__

        def __sub__(self, other):
            return FenicsVector(self.impl - other.impl)

        def __isub__(self, other):
            self._copy_data_if_needed()
            self.impl -= other.impl
            return self

        def __mul__(self, other):
            return FenicsVector(self.impl * other)

        def __neg__(self):
            return FenicsVector(-self.impl)

    class ComplexifiedFenicsVector(ComplexifiedVector):

        def amax(self):
            if self.imag_part is None:
                A = np.abs(self.real_part.impl.get_local())
            else:
                A = np.abs(self.real_part.impl.get_local() + self.imag_part.impl.get_local() * 1j)
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

    class FenicsVectorSpace(ComplexifiedListVectorSpace):

        complexified_vector_type = ComplexifiedFenicsVector

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

        def real_zero_vector(self):
            impl = df.Function(self.V).vector()
            return FenicsVector(impl)

        def real_full_vector(self, value):
            impl = df.Function(self.V).vector()
            impl += value
            return FenicsVector(impl)

        def real_random_vector(self, distribution, random_state, **kwargs):
            impl = df.Function(self.V).vector()
            values = _create_random_values(impl.local_size(), distribution, random_state, **kwargs)
            impl[:] = values
            return FenicsVector(impl)

        def real_make_vector(self, obj):
            return FenicsVector(obj)

    class FenicsMatrixOperator(LinearComplexifiedListVectorArrayOperatorBase):
        """Wraps a FEniCS matrix as an |Operator|."""

        def __init__(self, matrix, source_space, range_space, solver_options=None, name=None):
            assert matrix.rank() == 2
            self.__auto_init(locals())
            self.source = FenicsVectorSpace(source_space)
            self.range = FenicsVectorSpace(range_space)

        def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
            r = self.range.real_zero_vector()
            self.matrix.mult(u.impl, r.impl)
            return r

        def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
            r = self.source.real_zero_vector()
            self.matrix.transpmult(v.impl, r.impl)
            return r

        def _real_apply_inverse_one_vector(self, v, mu=None, least_squares=False, prepare_data=None):
            if least_squares:
                raise NotImplementedError
            r = self.source.real_zero_vector()
            options = self.solver_options.get('inverse') if self.solver_options else None
            _apply_inverse(self.matrix, r.impl, v.impl, options)
            return r

        def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
            if not all(isinstance(op, FenicsMatrixOperator) for op in operators):
                return None
            if identity_shift != 0:
                return None
            if np.iscomplexobj(coefficients):
                return None
            assert not solver_options

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
            else:
                matrix = operators[0].matrix * coefficients[0]
            for op, c in zip(operators[1:], coefficients[1:]):
                matrix.axpy(c, op.matrix, False)
                # in general, we cannot assume the same nonzero pattern for # all matrices. how to improve this?

            return FenicsMatrixOperator(matrix, self.source.V, self.range.V, name=name)

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
                    if u.imag_part is not None:
                        raise NotImplementedError
                    function.vector()[:] = u.real_part.impl
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
                        vec = u._list[0].real_part.impl
                        vmin = min(vmin, vec.min())
                        vmax = max(vmax, vec.max())

                for i, u in enumerate(U):
                    if u._list[0].imag_part is not None:
                        raise NotImplementedError
                    function = df.Function(self.space.V)
                    function.vector()[:] = u._list[0].real_part.impl
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
