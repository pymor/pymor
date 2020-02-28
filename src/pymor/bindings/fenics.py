# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_FENICS:
    import dolfin as df
    import ufl
    import numpy as np

    from pymor.core.base import BasicObject
    from pymor.core.defaults import defaults
    from pymor.operators.constructions import ZeroOperator
    from pymor.operators.interface import Operator
    from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.vectorarrays.interface import _create_random_values
    from pymor.vectorarrays.list import CopyOnWriteVector, ComplexifiedVector, ComplexifiedListVectorSpace
    from pymor.vectorarrays.numpy import NumpyVectorSpace

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
            return self.impl.get_local()  # always returns a copy

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
                max_global_ind_on_rank = max_ind_on_rank + self.real_part.impl.local_range()[0]
                comm = self.real_part.impl.mpi_comm()
                comm_size = comm.Get_size()

                max_inds = np.empty(comm_size, dtype='i')
                comm.Allgather(np.array(max_global_ind_on_rank, dtype='i'), max_inds)

                max_vals = np.empty(comm_size, dtype=np.float64)
                comm.Allgather(np.array(max_val_on_rank), max_vals)

                i = np.argmax(max_vals)
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
            impl[:] = np.ascontiguousarray(values)
            return FenicsVector(impl)

        def real_vector_from_numpy(self, data, ensure_copy=False):
            impl = df.Function(self.V).vector()
            impl[:] = np.ascontiguousarray(data)
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

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
            else:
                matrix = operators[0].matrix * coefficients[0]
            for op, c in zip(operators[1:], coefficients[1:]):
                matrix.axpy(c, op.matrix, False)
                # in general, we cannot assume the same nonzero pattern for # all matrices. how to improve this?

            return FenicsMatrixOperator(matrix, self.source.V, self.range.V, solver_options=solver_options, name=name)

    class FenicsOperator(Operator):
        """Wraps a FEniCS form as an |Operator|."""

        linear = False

        def __init__(self, form, source_space, range_space, source_function, dirichlet_bcs=(),
                     parameter_setter=None, parameter_type=None, solver_options=None, name=None):
            assert len(form.arguments()) == 1
            self.__auto_init(locals())
            self.source = source_space
            self.range = range_space
            self.build_parameter_type(parameter_type)

        def _set_mu(self, mu=None):
            mu = self.parse_parameter(mu)
            if self.parameter_setter:
                self.parameter_setter(mu)

        def apply(self, U, mu=None):
            assert U in self.source
            self._set_mu(mu)
            R = []
            source_vec = self.source_function.vector()
            for u in U._list:
                if u.imag_part is not None:
                    raise NotImplementedError
                source_vec[:] = u.real_part.impl
                r = df.assemble(self.form)
                for bc in self.dirichlet_bcs:
                    bc.apply(r, source_vec)
                R.append(r)
            return self.range.make_array(R)

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            if U._list[0].imag_part is not None:
                raise NotImplementedError
            self._set_mu(mu)
            source_vec = self.source_function.vector()
            source_vec[:] = U._list[0].real_part.impl
            matrix = df.assemble(df.derivative(self.form, self.source_function))
            for bc in self.dirichlet_bcs:
                bc.apply(matrix)
            return FenicsMatrixOperator(matrix, self.source.V, self.range.V)

        def restricted(self, dofs):
            from pymor.tools.mpi import parallel
            if parallel:
                raise NotImplementedError('SubMesh does not work in parallel')
            with self.logger.block(f'Restricting operator to {len(dofs)} dofs ...'):
                if len(dofs) == 0:
                    return ZeroOperator(NumpyVectorSpace(0), NumpyVectorSpace(0)), np.array([], dtype=np.int)

                if self.source.V.mesh().id() != self.range.V.mesh().id():
                    raise NotImplementedError

                self.logger.info('Computing affected cells ...')
                mesh = self.source.V.mesh()
                range_dofmap = self.range.V.dofmap()
                affected_cell_indices = set()
                for c in df.cells(mesh):
                    cell_index = c.index()
                    local_dofs = range_dofmap.cell_dofs(cell_index)
                    for ld in local_dofs:
                        if ld in dofs:
                            affected_cell_indices.add(cell_index)
                            continue
                affected_cell_indices = list(sorted(affected_cell_indices))

                if any(i.integral_type() not in ('cell', 'exterior_facet')
                       for i in self.form.integrals()):
                    # enlarge affected_cell_indices if needed
                    raise NotImplementedError

                self.logger.info('Computing source DOFs ...')
                source_dofmap = self.source.V.dofmap()
                source_dofs = set()
                for cell_index in affected_cell_indices:
                    local_dofs = source_dofmap.cell_dofs(cell_index)
                    source_dofs.update(local_dofs)
                source_dofs = np.array(sorted(source_dofs), dtype=np.intc)

                self.logger.info('Building submesh ...')
                subdomain = df.MeshFunction('size_t', mesh, mesh.geometry().dim())
                for ci in affected_cell_indices:
                    subdomain.set_value(ci, 1)
                submesh = df.SubMesh(mesh, subdomain, 1)

                self.logger.info('Building UFL form on submesh ...')
                form_r, V_r_source, V_r_range, source_function_r = self._restrict_form(submesh, source_dofs)

                self.logger.info('Building DirichletBCs on submesh ...')
                bc_r = self._restrict_dirichlet_bcs(submesh, source_dofs, V_r_source)

                self.logger.info('Computing source DOF mapping ...')
                restricted_source_dofs = self._build_dof_map(self.source.V, V_r_source, source_dofs)

                self.logger.info('Computing range DOF mapping ...')
                restricted_range_dofs = self._build_dof_map(self.range.V, V_r_range, dofs)

                op_r = FenicsOperator(form_r, FenicsVectorSpace(V_r_source), FenicsVectorSpace(V_r_range),
                                      source_function_r, dirichlet_bcs=bc_r, parameter_setter=self.parameter_setter,
                                      parameter_type=self.parameter_type)

                return (RestrictedFenicsOperator(op_r, restricted_range_dofs),
                        source_dofs[np.argsort(restricted_source_dofs)])

        def _restrict_form(self, submesh, source_dofs):
            V_r_source = df.FunctionSpace(submesh, self.source.V.ufl_element())
            V_r_range = df.FunctionSpace(submesh, self.range.V.ufl_element())
            assert V_r_source.dim() == len(source_dofs)

            if self.source.V != self.range.V:
                assert all(arg.ufl_function_space() != self.source.V for arg in self.form.arguments())
            args = tuple((df.function.argument.Argument(V_r_range, arg.number(), arg.part())
                          if arg.ufl_function_space() == self.range.V else arg)
                         for arg in self.form.arguments())

            if any(isinstance(coeff, df.Function) and coeff != self.source_function for coeff in
                   self.form.coefficients()):
                raise NotImplementedError

            source_function_r = df.Function(V_r_source)
            form_r = ufl.replace_integral_domains(
                self.form(*args, coefficients={self.source_function: source_function_r}),
                submesh.ufl_domain()
            )

            return form_r, V_r_source, V_r_range, source_function_r

        def _restrict_dirichlet_bcs(self, submesh, source_dofs, V_r_source):
            mesh = self.source.V.mesh()
            parent_facet_indices = compute_parent_facet_indices(submesh, mesh)

            def restrict_dirichlet_bc(bc):
                # ensure that markers are initialized
                bc.get_boundary_values()
                facets = np.zeros(mesh.num_facets(), dtype=np.uint)
                facets[bc.markers()] = 1
                facets_r = facets[parent_facet_indices]
                sub_domains = df.MeshFunction('size_t', submesh, mesh.topology().dim() - 1)
                sub_domains.array()[:] = facets_r

                bc_r = df.DirichletBC(V_r_source, bc.value(), sub_domains, 1, bc.method())
                return bc_r

            return tuple(restrict_dirichlet_bc(bc) for bc in self.dirichlet_bcs)

        def _build_dof_map(self, V, V_r, dofs):
            u = df.Function(V)
            u_vec = u.vector()
            restricted_dofs = []
            for dof in dofs:
                u_vec.zero()
                u_vec[dof] = 1
                u_r = df.interpolate(u, V_r)
                u_r = u_r.vector().get_local()
                if not np.all(np.logical_or(np.abs(u_r) < 1e-10, np.abs(u_r - 1.) < 1e-10)):
                    raise NotImplementedError
                r_dof = np.where(np.abs(u_r - 1.) < 1e-10)[0]
                if not len(r_dof) == 1:
                    raise NotImplementedError
                restricted_dofs.append(r_dof[0])
            restricted_dofs = np.array(restricted_dofs, dtype=np.int32)
            assert len(set(restricted_dofs)) == len(set(dofs))
            return restricted_dofs

    class RestrictedFenicsOperator(Operator):

        linear = False

        def __init__(self, op, restricted_range_dofs):
            self.source = NumpyVectorSpace(op.source.dim)
            self.range = NumpyVectorSpace(len(restricted_range_dofs))
            self.op = op
            self.restricted_range_dofs = restricted_range_dofs
            self.build_parameter_type(op)

        def apply(self, U, mu=None):
            assert U in self.source
            UU = self.op.source.zeros(len(U))
            for uu, u in zip(UU._list, U.data):
                uu.real_part.impl[:] = np.ascontiguousarray(u)
            VV = self.op.apply(UU, mu=mu)
            V = self.range.zeros(len(VV))
            for v, vv in zip(V.data, VV._list):
                v[:] = vv.real_part.impl[self.restricted_range_dofs]
            return V

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            UU = self.op.source.zeros()
            UU._list[0].real_part.impl[:] = np.ascontiguousarray(U.data[0])
            JJ = self.op.jacobian(UU, mu=mu)
            return NumpyMatrixOperator(JJ.matrix.array()[self.restricted_range_dofs, :])

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

    class FenicsVisualizer(BasicObject):
        """Visualize a FEniCS grid function.

        Parameters
        ----------
        space
            The `FenicsVectorSpace` for which we want to visualize DOF vectors.
        mesh_refinements
            Number of uniform mesh refinements to perform for vtk visualization
            (of functions from higher-order FE spaces).
        """

        def __init__(self, space, mesh_refinements=0):
            self.space = space
            self.mesh_refinements = mesh_refinements

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
                Filled in by :meth:`pymor.models.interface.Model.visualize` (ignored).
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
                coarse_function = df.Function(self.space.V)
                if self.mesh_refinements:
                    mesh = self.space.V.mesh()
                    for _ in range(self.mesh_refinements):
                        mesh = df.refine(mesh)
                    V_fine = df.FunctionSpace(mesh, self.space.V.ufl_element())
                    function = df.Function(V_fine)
                else:
                    function = coarse_function
                if legend:
                    function.rename(legend, legend)
                for u in U._list:
                    if u.imag_part is not None:
                        raise NotImplementedError
                    coarse_function.vector()[:] = u.real_part.impl
                    if self.mesh_refinements:
                        function.vector()[:] = df.interpolate(coarse_function, V_fine).vector()
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

    # adapted from dolfin.mesh.ale.init_parent_edge_indices
    def compute_parent_facet_indices(submesh, mesh):
        dim = mesh.topology().dim()
        facet_dim = dim - 1
        submesh.init(facet_dim)
        mesh.init(facet_dim)

        # Make sure we have vertex-facet connectivity for parent mesh
        mesh.init(0, facet_dim)

        parent_vertex_indices = submesh.data().array("parent_vertex_indices", 0)
        # Create the fact map
        parent_facet_indices = np.full(submesh.num_facets(), -1)

        # Iterate over the edges and figure out their parent number
        for local_facet in df.facets(submesh):

            # Get parent indices for edge vertices
            vs = local_facet.entities(0)
            Vs = [df.Vertex(mesh, parent_vertex_indices[int(v)]) for v in vs]

            # Get outgoing facets from the two parent vertices
            facets = [set(V.entities(facet_dim)) for V in Vs]

            # Check intersection
            common_facets = facets[0]
            for f in facets[1:]:
                common_facets = common_facets.intersection(f)
            assert len(common_facets) == 1
            parent_facet_index = list(common_facets)[0]

            # Set value
            parent_facet_indices[local_facet.index()] = parent_facet_index
        return parent_facet_indices
