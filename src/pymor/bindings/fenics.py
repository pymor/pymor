# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_FENICS:
    import dolfin as df
    import numpy as np

    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import OperatorBase
    from pymor.operators.constructions import ZeroOperator
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace

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
            x = df.Vector()
            self.impl.gather(x, dof_indices)
            return x.get_local()

        def amax(self):
            A = np.abs(self.impl.get_local())
            # there seems to be no way in the interface to compute amax without making a copy.
            # also, we need to check how things behave in the MPI parallel case.
            max_ind = np.argmax(A)
            max_val = A[max_ind]
            return max_ind, max_val

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

    class FenicsVectorSpace(ListVectorSpace):

        def __init__(self, V, id_='STATE'):
            self.V = V
            self.id = id_

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

        def make_vector(self, obj):
            return FenicsVector(obj)

    class FenicsMatrixOperator(OperatorBase):
        """Wraps a FEniCS matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, source_space, range_space, solver_options=None, name=None):
            assert matrix.rank() == 2
            self.source = FenicsVectorSpace(source_space)
            self.range = FenicsVectorSpace(range_space)
            self.matrix = matrix
            self.solver_options = solver_options
            self.name = name

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                self.matrix.mult(u.impl, r.impl)
            return R

        def apply_adjoint(self, V, mu=None):
            assert V in self.range
            U = self.source.zeros(len(V))
            for v, u in zip(V._list, U._list):
                self.matrix.transpmult(v.impl, u.impl)  # there are no complex numbers in FEniCS
            return U

        def apply_inverse(self, V, mu=None, least_squares=False):
            assert V in self.range
            if least_squares:
                raise NotImplementedError
            R = self.source.zeros(len(V))
            options = self.solver_options.get('inverse') if self.solver_options else None
            for r, v in zip(R._list, V._list):
                _apply_inverse(self.matrix, r.impl, v.impl, options)
            return R

        def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
            if not all(isinstance(op, (FenicsMatrixOperator, ZeroOperator)) for op in operators):
                return None
            assert not solver_options

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
            else:
                matrix = operators[0].matrix * coefficients[0]
            for op, c in zip(operators[1:], coefficients[1:]):
                if isinstance(op, ZeroOperator):
                    continue
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

        def visualize(self, U, d, title='', legend=None, filename=None, block=True,
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
            d
                Filled in by :meth:`pymor.discretizations.DiscretizationBase.visualize` (ignored).
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
