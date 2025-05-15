# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('FENICSX')


import numpy as np
from dolfinx.fem import Constant, Function
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.la import create_petsc_vector
from dolfinx.plot import vtk_mesh
from petsc4py import PETSc

from pymor.core.base import ImmutableObject
from pymor.core.defaults import defaults
from pymor.core.pickle import unpicklable
from pymor.operators.constructions import MutableState, MutableStateOperator, VectorFunctional, VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
from pymor.vectorarrays.interface import _create_random_values
from pymor.vectorarrays.list import ComplexifiedListVectorSpace, ComplexifiedVector, CopyOnWriteVector
from pymor.vectorarrays.numpy import NumpyVectorSpace


@unpicklable
class FenicsxVector(CopyOnWriteVector):
    """Wraps a FEniCSx vector to make it usable with ListVectorArray."""

    def __init__(self, impl):
        self.impl = impl

    @classmethod
    def from_instance(cls, instance):
        return cls(instance.impl)

    def _copy_data(self):
        self.impl = self.impl.copy()

    def to_numpy(self, ensure_copy=False):
        return self.impl.array.copy() if ensure_copy else self.impl.array  # TODO: what happens here in parallel)

    def _scal(self, alpha):
        self.impl *= alpha

    def _axpy(self, alpha, x):
        if x is self:
            self.scal(1. + alpha)
        else:
            self.impl.axpy(alpha, x.impl)

    def inner(self, other):
        return self.impl.dot(other.impl)

    def norm(self):
        return self.impl.norm(PETSc.NormType.NORM_2)  # TODO: parallel?

    def norm2(self):
        return self.impl.norm(PETSc.NormType.NORM_2) ** 2

    def sup_norm(self):
        return self.impl.norm(PETSc.NormType.NORM_INFINITY)  # TODO: parallel?

    def dofs(self, dof_indices):
        dof_indices = np.array(dof_indices, dtype=np.intc)
        if len(dof_indices) == 0:
            return np.array([], dtype=np.intc)
        return self.imp.getValues(dof_indices)  # TODO: Global indices but only for local processor allowd

    def amax(self):
        raise NotImplementedError  # is implemented for complexified vector

    def __add__(self, other):
        return FenicsxVector(self.impl + other.impl)

    def __iadd__(self, other):
        self._copy_data_if_needed()
        self.impl += other.impl
        return self

    __radd__ = __add__

    def __sub__(self, other):
        return FenicsxVector(self.impl - other.impl)

    def __isub__(self, other):
        self._copy_data_if_needed()
        self.impl -= other.impl
        return self

    def __mul__(self, other):
        return FenicsxVector(self.impl * other)

    def __neg__(self):
        return FenicsxVector(-self.impl)


class ComplexifiedFenicsxVector(ComplexifiedVector):

    def amax(self):
        raise NotImplementedError


class FenicsxVectorSpace(ComplexifiedListVectorSpace):

    real_vector_type = FenicsxVector
    vector_type = ComplexifiedFenicsxVector

    def __init__(self, V):
        self.__auto_init(locals())

    @property
    def dim(self):
        return self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs

    def __eq__(self, other):
        return type(other) is FenicsxVectorSpace \
            and getattr(self.V, '_cpp_object', self.V) == getattr(other.V, '_cpp_object', other.V)

    def __hash__(self):
        return id(self.V)

    def real_zero_vector(self):
        impl = create_petsc_vector(self.V.dofmap.index_map, self.V.dofmap.index_map_bs)
        return FenicsxVector(impl)

    def real_full_vector(self, value):
        v = self.real_zero_vector()
        v.impl.set(value)
        return v

    def real_random_vector(self, distribution, random_state, **kwargs):
        v = self.real_zero_vector()
        values = _create_random_values(self.dim, distribution, random_state, **kwargs)  # TODO: parallel?
        v.to_numpy()[:] = values
        return v

    def real_vector_from_numpy(self, data, ensure_copy=False):
        v = self.real_zero_vector()
        v.to_numpy()[:] = data
        return v

    def real_make_vector(self, obj):
        return FenicsxVector(obj)


class FenicsxMatrixBasedOperator(Operator):
    """Wraps a parameterized FEniCSx linear or bilinear form as an |Operator|.

    Parameters
    ----------
    form
        The `Form` object which is assembled to a matrix or vector.
    params
        Dict mapping parameters to dolfinx `Constants`.
    bcs
        dolfin `DirichletBC` objects to be applied.
    functional
        If `True` return a |VectorFunctional| instead of a |VectorOperator| in case
        `form` is a linear form.
    solver_options
        The |solver_options| for the assembled :class:`FenicsMatrixOperator`.
    name
        Name of the operator.
    """

    linear = True

    def __init__(self, form, params=None, bcs=None, lifting_form=None, functional=False, solver_options=None,
                 name=None):
        assert 1 <= form.rank <= 2
        params = params or {}
        bcs = bcs or tuple()
        assert all(isinstance(v, Constant) and len(v.ufl_shape) <= 1 for v in params.values())
        assert not functional or form.rank == 1
        self.__auto_init(locals())
        if form.rank == 2 or not functional:
            range_space = form.function_spaces[0]
            self.range = FenicsxVectorSpace(range_space)
        else:
            self.range = NumpyVectorSpace(1)
        if form.rank == 2 or functional:
            source_space = form.function_spaces[0 if functional else 1]  # TODO: check order
            self.source = FenicsxVectorSpace(source_space)
        else:
            self.source = NumpyVectorSpace(1)
        self.parameters_own = {k: v.ufl_shape[0] if len(v.ufl_shape) == 1 else 1 for k, v in params.items()}

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        # update coefficients in form
        for k, v in self.params.items():
            v.value = mu[k]
        # assemble matrix
        if self.form.rank == 2:
            mat = assemble_matrix(self.form, self.bcs)
            mat.assemble()
            return FenicsxMatrixOperator(mat, self.range.V, self.source.V, self.solver_options,
                                         self.name + '_assembled')
        elif self.functional:
            vec = assemble_vector(self.form)
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            V = self.source.make_array([vec])
            return VectorFunctional(V)
        else:
            vec = assemble_vector(self.form)
            apply_lifting(vec, [self.lifting_form], [self.bcs])
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(vec, self.bcs)
            V = self.range.make_array([vec])
            return VectorOperator(V)

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(V)

    def as_range_array(self, mu=None):
        return self.assemble(mu).as_range_array()

    def as_source_array(self, mu=None):
        return self.assemble(mu).as_source_array()

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return self.assemble(mu).apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)


class FenicsxMatrixOperator(LinearComplexifiedListVectorArrayOperatorBase):
    """Wraps a FEniCSx matrix as an |Operator|."""

    def __init__(self, matrix, range_space, source_space, solver_options=None, name=None):
        self.__auto_init(locals())
        self.range = FenicsxVectorSpace(range_space)
        self.source = FenicsxVectorSpace(source_space)

    def _solver_options(self, adjoint=False):
        if adjoint:
            options = self.solver_options.get('inverse_adjoint') if self.solver_options else None
            if options is None:
                options = self.solver_options.get('inverse') if self.solver_options else None
        else:
            options = self.solver_options.get('inverse') if self.solver_options else None
        return options or _solver_options()

    def _create_solver(self, adjoint=False):
        options = self._solver_options(adjoint)
        if adjoint:
            try:
                matrix = self._matrix_transpose
            except AttributeError as e:
                raise RuntimeError('_create_solver called before _matrix_transpose has been initialized.') from e
        else:
            matrix = self.matrix
        method = options.get('solver')
        preconditioner = options.get('preconditioner')
        solver = PETSc.KSP().create(self.source.V.mesh.comm)
        solver.setOperators(matrix)
        solver.setType(method)
        solver.getPC().setType(preconditioner)
        return solver

    def _apply_inverse(self, r, v, adjoint=False):
        try:
            solver = self._adjoint_solver if adjoint else self._solver
        except AttributeError:
            solver = self._create_solver(adjoint)
        solver.solve(v, r)  # TODO: might overwrite v
        if _solver_options()['keep_solver']:
            if adjoint:
                self._adjoint_solver = solver
            else:
                self._solver = solver

    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        r = self.range.real_zero_vector()
        self.matrix.mult(u.impl, r.impl)
        return r

    def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        r = self.source.real_zero_vector()
        self.matrix.multTranspose(v.impl, r.impl)
        return r

    def _real_apply_inverse_one_vector(self, v, mu=None, initial_guess=None,
                                       least_squares=False, prepare_data=None):
        if least_squares:
            raise NotImplementedError
        r = (self.source.real_zero_vector() if initial_guess is None else
             initial_guess.copy(deep=True))
        self._apply_inverse(r.impl, v.impl)
        return r

    def _real_apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None,
                                               least_squares=False, prepare_data=None):
        if least_squares:
            raise NotImplementedError
        r = (self.range.real_zero_vector() if initial_guess is None else
             initial_guess.copy(deep=True))

        # since dolfin does not have "apply_inverse_adjoint", we assume
        # PETSc is used as backend and transpose the matrix
        if not hasattr(self, '_matrix_transpose'):
            self._matrix_transpose = PETSc.Mat()
            self.matrix.transpose(self._matrix_transpose)
        self._apply_inverse(r.impl, u.impl, adjoint=True)
        return r

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
        if not all(isinstance(op, FenicsxMatrixOperator) for op in operators):
            return None
        if identity_shift != 0:
            return None
        if np.iscomplexobj(coefficients):
            return None

        if coefficients[0] == 1:
            matrix = operators[0].matrix.copy()
        else:
            matrix = operators[0].matrix * coefficients[0]
        for op, c in zip(operators[1:], coefficients[1:]):
            matrix.axpy(c, op.matrix)
            # in general, we cannot assume the same nonzero pattern for
            # all matrices. how to improve this?

        return FenicsxMatrixOperator(matrix, self.source.V, self.range.V, solver_options=solver_options, name=name)


@defaults('solver', 'preconditioner', 'keep_solver')
def _solver_options(solver=PETSc.KSP.Type.PREONLY,
                    preconditioner=PETSc.PC.Type.LU, keep_solver=True):
    return {'solver': solver, 'preconditioner': preconditioner, 'keep_solver': keep_solver}


class FenicsxVisualizer(ImmutableObject):
    """Visualize a FEniCSx grid function.

    Parameters
    ----------
    space
        The `FenicsVectorSpace` for which we want to visualize DOF vectors.
    """

    def __init__(self, space):
        self.space = space

    def visualize(self, U, title='', legend=None, filename=None, block=True,
                  separate_colorbars=True):
        """Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize (length must be 1). Alternatively,
            a tuple of |VectorArrays| which will be visualized in separate windows.
            If `filename` is specified, only one |VectorArray| may be provided which,
            however, is allowed to contain multiple vectors that will be interpreted
            as a time series.
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
            raise NotImplementedError
        else:
            assert U in self.space and len(U) == 1 \
                or (isinstance(U, tuple) and all(u in self.space for u in U) and all(len(u) == 1 for u in U))
            if not isinstance(U, tuple):
                U = (U,)
            if isinstance(legend, str):
                legend = (legend,)
            if legend is None:
                legend = tuple(f'U{i}' for i in range(len(U)))
            assert legend is None or len(legend) == len(U)

            import pyvista
            rows = 1 if len(U) <= 2 else 2
            cols = int(np.ceil(len(U) / rows))
            plotter = pyvista.Plotter(shape=(rows, cols))
            mesh_data = vtk_mesh(self.space.V)
            for i, (u, l) in enumerate(zip(U, legend)):
                row = i // cols
                col = i - row*cols
                plotter.subplot(row, col)
                u_grid = pyvista.UnstructuredGrid(*mesh_data)
                u_grid.point_data[l] = u.vectors[0].real_part.impl.array.real
                u_grid.set_active_scalars(l)
                plotter.add_mesh(u_grid, show_edges=False)
                plotter.add_scalar_bar(l)
                plotter.view_xy()
            plotter.show()


class FenicsxMutableState(MutableState):
    def __init__(self, f):
        self.f = f
        super().__init__(FenicsxVectorSpace(f.function_space))

    def _set(self, state):
        assert state.vectors[0].imag_part is None
        with state.vectors[0].real_part.impl.localForm() as loc_state, self.f.x.petsc_vec.localForm() as loc_f:
            loc_state.copy(loc_f)


class FenicsxMutableStateMatrixBasedOperator(MutableStateOperator):

    linear_in_op_source = True
    _last_mu = None

    def __init__(self, mutable_states, form, params,
                 bcs=None, lifting_form=None, functional=False, solver_options=None, name=None):
        self._matrix_based_op = FenicsxMatrixBasedOperator(
            form, params,
            bcs=bcs, lifting_form=lifting_form, functional=functional, solver_options=solver_options, name=name)
        super().__init__(mutable_states, self._matrix_based_op.range, self._matrix_based_op.source)
        self.__auto_init(locals())
        self.parameters_own = self._matrix_based_op.parameters
        mutable_states = tuple(mutable_states)
        assert all(isinstance(ms, MutableState) for ms in mutable_states)

    def _assemble_matrix_if_needed(self, mu=None):
        if self._state_changed or mu is not self._last_mu:
            self._matrix_op = self._matrix_based_op.assemble(mu)
            self._last_mu = mu

    def _apply(self, U, mu=None):
        self._assemble_matrix_if_needed(mu=mu)
        return self._matrix_op.apply(U)

    def _apply_adjoint(self, V, mu=None):
        self._assemble_matrix_if_needed(mu=mu)
        return self._matrix_op.apply_adjoint(V)

    def _apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        self._assemble_matrix_if_needed(mu=mu)
        return self._matrix_op.apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)

    def _apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        self._assemble_matrix_if_needed(mu=mu)
        return self._matrix_op.apply_inverse_adjoint(U, initial_guess=initial_guess, least_squares=least_squares)

    def _jacobian(self, U, mu=None):
        self._assemble_matrix_if_needed(mu=mu)
        return self._matrix_op


class FenicsxInterpolationOperator(Operator):

    linear = True
    source = NumpyVectorSpace(1)

    def __init__(self, V, function, parameters):
        self.__auto_init(locals())
        self.range = FenicsxVectorSpace(V)
        self.parameters_own = parameters

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        f = Function(self.V)
        self.function.set_mu(mu)
        f.interpolate(self.function)
        # TODO: copy needed as petsc_vec does ensure that memory stays allocated
        return VectorOperator(self.range.make_array([f.x.petsc_vec.copy()]))

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(V)

    def as_range_array(self, mu=None):
        return self.assemble(mu).as_range_array()

    def as_source_array(self, mu=None):
        return self.assemble(mu).as_source_array()

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return self.assemble(mu).apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)
