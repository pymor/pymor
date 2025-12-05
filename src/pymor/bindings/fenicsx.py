# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('FENICSX')

from pymor.tools.mpi import parallel

if parallel:
    import warnings
    warnings.warn('MPI parallel run detected. FEniCSx bindings have only been tested for serial execution.')

import sys

import numpy as np
from dolfinx.fem import Constant, Function, IntegralType, create_interpolation_data, dirichletbc, form, functionspace
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, create_vector, set_bc
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from petsc4py import PETSc
from ufl import Argument, Form, derivative, replace

from pymor.core.base import ImmutableObject
from pymor.core.defaults import defaults
from pymor.core.pickle import unpicklable
from pymor.operators.constructions import VectorFunctional, VectorOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.solvers.list import ComplexifiedListVectorArrayBasedSolver
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
        return self.impl.getValues(dof_indices)  # TODO: Global indices but only for local processor allowd

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
        A = np.abs(self.to_numpy())
        max_ind = np.argmax(A)
        max_val = A[max_ind]
        return max_ind, max_val


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
        impl = create_vector(self.V)
        return FenicsxVector(impl)

    def real_full_vector(self, value):
        v = self.real_zero_vector()
        v.impl.set(value)
        return v

    def real_random_vector(self, distribution, **kwargs):
        v = self.real_zero_vector()
        values = _create_random_values(self.dim, distribution, **kwargs)  # TODO: parallel?
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
    ufl_form
        The ufl `Form` object which is assembled to a matrix or vector.
    params
        Dict mapping parameters to dolfinx `Constants`.
    bcs
        dolfinx `DirichletBC` objects to be applied.
    diag
        Value to put on the diagonal when applying Dirichlet conditions.
    ufl_lifting_form
        If not `None` and `ufl_form` represents a linear form, `apply_lifting` is called
        with this form for the assembled vector.
    functional
        If `True` and `ufl_form` represents a linear form, return a |VectorFunctional| instead
        of a |VectorOperator|.
    solver
        The |Solver| for the operator.
    name
        Name of the operator.
    """

    linear = True

    def __init__(self, ufl_form, params=None, bcs=(), diag=1., ufl_lifting_form=None, alpha=1., functional=False,
                 solver=None, name=None):
        rank = len(ufl_form.arguments())
        assert 1 <= rank <= 2
        params = params or {}
        assert all(isinstance(v, Constant) and len(v.ufl_shape) <= 1 for v in params.values())
        assert not functional or rank == 1
        if rank == 2 and [arg.number() for arg in ufl_form.arguments()] != [0, 1]:
            raise NotImplementedError
        self.__auto_init(locals())
        self.compiled_form = form(ufl_form)
        self.compiled_lifting_form = form(ufl_lifting_form)
        self.rank = rank
        if rank == 2 or not functional:
            self.range = FenicsxVectorSpace(ufl_form.arguments()[0].ufl_function_space())
        else:
            self.range = NumpyVectorSpace(1)
        if rank == 2 or functional:
            self.source = FenicsxVectorSpace(ufl_form.arguments()[-1].ufl_function_space())
        else:
            self.source = NumpyVectorSpace(1)
        self.parameters_own = {k: v.ufl_shape[0] if len(v.ufl_shape) == 1 else 1 for k, v in params.items()}

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        # update coefficients in form
        for k, v in self.params.items():
            v.value = mu[k]
        # assemble matrix
        if self.rank == 2:
            mat = assemble_matrix(self.compiled_form, bcs=self.bcs, diag=self.diag)
            mat.assemble()
            return FenicsxMatrixOperator(mat, self.range.V, self.source.V, solver=self.solver,
                                         name=self.name + '_assembled')
        else:
            vec = assemble_vector(self.compiled_form)
            if self.bcs and self.lifting_form:
                apply_lifting(vec, [self.compiled_lifting_form], [self.bcs], alpha=self.alpha)
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            if self.bcs:
                set_bc(vec, self.bcs, alpha=self.alpha)
            if self.functional:
                V = self.source.make_array([vec])
                return VectorFunctional(V)
            else:
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


class FenicsxLinearSolver(ComplexifiedListVectorArrayBasedSolver):

    @defaults('method', 'preconditioner', 'keep_solver')
    def __init__(self, comm, method=PETSc.KSP.Type.PREONLY, preconditioner=PETSc.PC.Type.LU, keep_solver=True):
        self.__auto_init(locals())
        if keep_solver:  # not thread safe
            self._solver = self._create_solver()
            self._adjoint_solver = self._create_solver()
            self._last_operator = None
            self._last_adjoint_operator = None

    def _create_solver(self):
        method, preconditioner = self.method, self.preconditioner
        solver = PETSc.KSP().create(self.comm)
        solver.setType(method)
        solver.getPC().setType(preconditioner)
        return solver

    def _prepare(self, operator, U, mu, adjoint):
        operator = operator.assemble(mu)

        if adjoint and not hasattr(operator, '_matrix_transpose'):
            # since dolfin does not have "apply_inverse_adjoint", we assume
            # PETSc is used as backend and transpose the matrix
            operator._matrix_transpose = PETSc.Mat()
            operator.matrix.transpose(operator._matrix_transpose)

        if self.keep_solver:
            if adjoint:
                solver = self._adjoint_solver
                if operator.uid != self._last_adjoint_operator:
                    solver.setOperators(operator._matrix_transpose)
                    self._last_adjoint_operator = operator.uid
            else:
                solver = self._solver
                if operator.uid != self._last_operator:
                    solver.setOperators(operator.matrix)
                    self._last_operator = operator.uid
        else:
            solver = self._create_solver()
            solver.setOperators(operator._matrix_transpose if adjoint else operator.matrix)
        return solver

    def _real_solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        solver = prepare_data
        r = (operator.source.real_zero_vector() if initial_guess is None else
             initial_guess.copy(deep=True))
        solver.setInitialGuessNonzero(initial_guess is not None)
        solver.solve(v.impl, r.impl)
        return r

    def _real_solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        solver = prepare_data
        r = (operator.range.real_zero_vector() if initial_guess is None else
             initial_guess.copy(deep=True))
        solver.setInitialGuessNonzero(initial_guess is not None)
        solver.solve(u.impl, r.impl)
        return r


class FenicsxMatrixOperator(LinearComplexifiedListVectorArrayOperatorBase):
    """Wraps a FEniCSx matrix as an |Operator|."""

    def __init__(self, matrix, range_space, source_space, solver=None, name=None):
        solver = solver or FenicsxLinearSolver(source_space.mesh.comm)
        self.__auto_init(locals())
        self.range = FenicsxVectorSpace(range_space)
        self.source = FenicsxVectorSpace(source_space)

    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        r = self.range.real_zero_vector()
        self.matrix.mult(u.impl, r.impl)
        return r

    def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        r = self.source.real_zero_vector()
        self.matrix.multTranspose(v.impl, r.impl)
        return r

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., name=None):
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
        for op, c in zip(operators[1:], coefficients[1:], strict=True):
            matrix.axpy(c, op.matrix)
            # in general, we cannot assume the same nonzero pattern for
            # all matrices. how to improve this?

        return FenicsxMatrixOperator(matrix, self.source.V, self.range.V, name=name)


class FenicsxOperator(Operator):
    """Wraps an UFL form as an |Operator|."""

    def __init__(self, ufl_form, source_function, params=None, bcs=(), alpha=None, linear=False,
                 apply_lifting_with_jacobian=False, solver=None, name=None):
        assert len(ufl_form.arguments()) == 1
        params = params or {}
        if alpha is None:
            alpha = -1 if apply_lifting_with_jacobian else 1
        assert all(isinstance(v, Constant) and len(v.ufl_shape) <= 1 for v in params.values())
        self.__auto_init(locals())
        self.range = FenicsxVectorSpace(ufl_form.arguments()[0].ufl_function_space())
        self.source = FenicsxVectorSpace(source_function.ufl_function_space())
        self.compiled_form = form(ufl_form)
        self.compiled_derivative = form(derivative(self.ufl_form, self.source_function))
        self.parameters_own = {k: v.ufl_shape[0] if len(v.ufl_shape) == 1 else 1 for k, v in params.items()}

    def _set_mu(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        for k, v in self.params.items():
            v.value = mu[k]

    def _set_source_function(self, U):
        assert len(U) == 1
        assert U.vectors[0].imag_part is None
        with (U.vectors[0].real_part.impl.localForm() as loc_u,
              self.source_function.x.petsc_vec.localForm() as loc_source_func):
            loc_u.copy(loc_source_func)

    def apply(self, U, mu=None):
        assert U in self.source
        self._set_mu(mu)
        R = []
        for u in U:
            self._set_source_function(u)
            vec = assemble_vector(self.compiled_form)
            if self.apply_lifting_with_jacobian:
                apply_lifting(vec, [self.compiled_derivative], bcs=[self.bcs], x0=[self.source_function.x.petsc_vec],
                              alpha=self.alpha)
                vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            if self.bcs:
                set_bc(vec, self.bcs, x0=self.source_function.x.petsc_vec, alpha=self.alpha)
            R.append(vec)
        return self.range.make_array(R)

    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1
        self._set_mu(mu)
        self._set_source_function(U)
        mat = assemble_matrix(self.compiled_derivative, self.bcs)
        mat.assemble()
        return FenicsxMatrixOperator(mat, self.range.V, self.source.V, solver=self._jacobian_solver,
                                     name=self.name + '_jacobian')

    def restricted(self, dofs):
        from pymor.tools.mpi import parallel
        if parallel:
            raise NotImplementedError
        with self.logger.block(f'Restricting operator to {len(dofs)} dofs ...'):
            if len(dofs) == 0:
                return ZeroOperator(NumpyVectorSpace(0), NumpyVectorSpace(0)), np.array([], dtype=int)

            if self.source.V.mesh != self.range.V.mesh:
                assert False
                raise NotImplementedError

            self.logger.info('Computing affected cells ...')
            mesh = self.source.V.mesh
            range_dofmap = self.range.V.dofmap
            affected_cells = set()
            num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            for c in range(num_cells):
                local_dofs = range_dofmap.cell_dofs(c)
                for ld in local_dofs:
                    if ld in dofs:
                        affected_cells.add(c)
                        continue
            affected_cells = sorted(affected_cells)

            if not self.compiled_form.integral_types <= {IntegralType.cell, IntegralType.exterior_facet}:
                # enlarge affected_cell_indices if needed
                raise NotImplementedError

            self.logger.info('Computing source DOFs ...')
            source_dofmap = self.source.V.dofmap
            source_dofs = set()
            for c in affected_cells:
                local_dofs = source_dofmap.cell_dofs(c)
                source_dofs.update(local_dofs)
            source_dofs = np.array(sorted(source_dofs), dtype=np.intc)

            self.logger.info('Building submesh ...')
            submesh, _, _, _ = create_submesh(mesh, mesh.topology.dim, np.array(affected_cells))

            self.logger.info('Building UFL form on submesh ...')
            form_r, source_function_r, params = self._restrict_form(submesh)
            V_r_source = source_function_r.ufl_function_space()
            V_r_range = form_r.arguments()[0].ufl_function_space()

            self.logger.info('Computing source DOF mapping ...')
            restricted_source_dofs = self._build_dof_map(self.source.V, V_r_source, source_dofs)

            self.logger.info('Computing range DOF mapping ...')
            restricted_range_dofs = self._build_dof_map(self.range.V, V_r_range, dofs)

            self.logger.info('Building DirichletBCs on submesh ...')
            bc_r = self._restrict_dirichlet_bcs(self.source.V, V_r_source, self.bcs, source_dofs,
                                                restricted_source_dofs)

            op_r = FenicsxOperator(form_r, source_function_r, params=params, bcs=bc_r, alpha=self.alpha,
                                   linear=self.linear, apply_lifting_with_jacobian=self.apply_lifting_with_jacobian)

            return (RestrictedFenicsxOperator(op_r, restricted_range_dofs),
                    source_dofs[np.argsort(restricted_source_dofs)])

    def _restrict_form(self, submesh):
        V_r_source = functionspace(submesh, self.source.V.ufl_element())
        if self.source == self.range:
            V_r_range = V_r_source
        else:
            assert False
            V_r_range = functionspace(submesh, self.range.V.ufl_element())
        assert V_r_source.dofmap.index_map.size_global * V_r_source.dofmap.index_map_bs

        if len(self.ufl_form.ufl_domains()) != 1:
            assert False
            raise NotImplementedError

        assert len(self.ufl_form.arguments()) == 1
        orig_args = self.ufl_form.arguments()
        args = tuple(Argument(V_r_range, arg.number(), arg.part()) for arg in orig_args)

        assert len(self.ufl_form.coefficients()) == 1
        orig_coeffs = self.ufl_form.coefficients()
        coeffs = (Function(V_r_source),)

        form_r = replace(self.ufl_form,
                         dict(zip(orig_args, args, strict=True)) | dict(zip(orig_coeffs, coeffs, strict=True)))

        integrals = [i.reconstruct(domain=submesh.ufl_domain()) for i in form_r.integrals()]
        form_r = Form(integrals)

        # Reusing the original ufl Constant, even though it is defined over the original mesh
        # seems to work, even though redefining the form with the old Constant fails.
        params = self.params

        return form_r, coeffs[0], params

    @staticmethod
    def _restrict_dirichlet_bcs(V, V_r, bcs, source_dofs, restricted_source_dofs):
        u = Function(V)
        U = u.x.array

        restricted_bcs = []
        for bc in bcs:
            bc_dofs = bc.dof_indices()[0]
            U[:] = 0.
            bc.set(U)
            u_r = Function(V_r)
            u_r.x.array[restricted_source_dofs] = U[source_dofs]
            dofs = []
            for sd, rsd in zip(source_dofs, restricted_source_dofs, strict=True):
                if sd in bc_dofs:
                    dofs.append(rsd)
            if dofs:
                restricted_bcs.append(dirichletbc(u_r, np.array(dofs)))

        return tuple(restricted_bcs)

    @staticmethod
    def _build_dof_map(V, V_r, dofs):
        submesh = V_r.mesh
        cells = np.arange(submesh.topology.index_map(submesh.topology.dim).size_local)
        u = Function(V)
        u_r = Function(V_r)
        u.x.array[dofs] = np.arange(1, len(dofs)+1)
        if V.num_sub_spaces > 0:
            for i in range(V.num_sub_spaces):
                interpolation_data = create_interpolation_data(V_r.sub(i), V.sub(i), cells)
                u_r.sub(i).interpolate_nonmatching(u.sub(i), cells, interpolation_data=interpolation_data)
        else:
            interpolation_data = create_interpolation_data(V_r, V, cells)
            u_r.interpolate_nonmatching(u, cells, interpolation_data=interpolation_data)
        sorted_ind = np.argsort(u_r.x.array)
        u_r_sorted = u_r.x.array[sorted_ind]
        if abs(u_r_sorted[0]) > 1e-7:
            first_nonzero = None
            if len(u_r_sorted) != len(dofs):
                raise NotImplementedError
        else:
            first_nonzero = (u_r_sorted > (1. - 1e-7)).nonzero()[0][0]
            if len(u_r_sorted) - first_nonzero != len(dofs):
                raise NotImplementedError
        if not np.all(np.abs(u_r_sorted[first_nonzero:] - np.arange(1, len(dofs)+1)) < 1e-7):
            raise NotImplementedError
        restricted_dofs = sorted_ind[first_nonzero:]
        return restricted_dofs


class RestrictedFenicsxOperator(Operator):
    """Restricted :class:`FenicsxOperator`."""

    linear = False

    def __init__(self, op, restricted_range_dofs, solver=None):
        self.__auto_init(locals())
        self.source = NumpyVectorSpace(op.source.dim)
        self.range = NumpyVectorSpace(len(restricted_range_dofs))

    def apply(self, U, mu=None):
        assert U in self.source
        UU = self.op.source.zeros(len(U))
        for uu, u in zip(UU.vectors, U.to_numpy().T, strict=True):
            uu.real_part.impl[:] = np.ascontiguousarray(u)
        VV = self.op.apply(UU, mu=mu)
        V = VV.to_numpy()[self.restricted_range_dofs, :]
        return self.range.from_numpy(V)

    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1
        UU = self.op.source.zeros()
        UU.vectors[0].real_part.impl[:] = np.ascontiguousarray(U.to_numpy()[:, 0])
        JJ = self.op.jacobian(UU, mu=mu)
        dense_mat = JJ.matrix.convert('dense')
        array = dense_mat.getDenseArray()
        return NumpyMatrixOperator(array[self.restricted_range_dofs, :],
                                   solver=self._jacobian_solver)


class FenicsxVisualizer(ImmutableObject):
    """Visualize a FEniCSx grid function.

    Parameters
    ----------
    space
        The `FenicsxVectorSpace` for which we want to visualize DOF vectors.
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
            called_from_test = getattr(sys, '_called_from_test', False)
            plotter = pyvista.Plotter(shape=(rows, cols), off_screen=called_from_test)
            mesh_data = vtk_mesh(self.space.V)
            for i, (u, l) in enumerate(zip(U, legend, strict=True)):
                row = i // cols
                col = i - row*cols
                plotter.subplot(row, col)
                u_grid = pyvista.UnstructuredGrid(*mesh_data)
                u_grid.point_data[l] = u.vectors[0].real_part.impl.array.real
                u_grid.set_active_scalars(l)
                plotter.add_mesh(u_grid, show_edges=False)
                plotter.add_scalar_bar(l)
                plotter.view_xy()
                plotter.add_title(title)
            plotter.show()
