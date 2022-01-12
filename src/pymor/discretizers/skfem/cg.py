# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('SCIKIT_FEM')


import numpy as np
from skfem import Basis, BoundaryFacetBasis, BilinearForm, LinearForm, asm, enforce, projection
from skfem.helpers import grad, dot
from skfem.visuals.matplotlib import plot, show

from pymor.algorithms.preassemble import preassemble as preassemble_
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction
from pymor.core.base import ImmutableObject
from pymor.discretizers.skfem.domaindiscretizer import discretize_domain
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class SKFemBilinearFormOperator(NumpyMatrixBasedOperator):

    sparse = True

    def __init__(self, basis, dirichlet_dofs=None, dirichlet_clear_diag=False, name=None):
        self.source = self.range = NumpyVectorSpace(basis.N)
        self.__auto_init(locals())

    def build_form(mu):
        pass

    def _assemble(self, mu):
        form = BilinearForm(self.build_form(mu))
        A = asm(form, self.basis)
        if self.dirichlet_dofs is not None:
            A.setdiag(A.diagonal())
            enforce(A, D=self.dirichlet_dofs, diag=0. if self.dirichlet_clear_diag else 1., overwrite=True)
        return A


class SKFemLinearFormOperator(NumpyMatrixBasedOperator):

    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, basis, dirichlet_dofs=None, name=None):
        self.range = NumpyVectorSpace(basis.N)
        self.__auto_init(locals())

    def _assemble(self, mu):
        form = LinearForm(self.build_form(mu))
        F = asm(form, self.basis)
        if self.dirichlet_dofs is not None:
            F[self.dirichlet_dofs] = 0
        return F.reshape((-1, 1))


class DiffusionOperator(SKFemBilinearFormOperator):

    def __init__(self, basis, diffusion_function, dirichlet_dofs=None, dirichlet_clear_diag=False, name=None):
        super().__init__(basis, dirichlet_dofs=dirichlet_dofs, dirichlet_clear_diag=dirichlet_clear_diag, name=name)
        self.__auto_init(locals())

    def build_form(self, mu):
        def bf(u, v, w):
            d = _eval_pymor_function(self.diffusion_function, w.x, mu)
            return dot(grad(u), grad(v)) * d
        return bf


class L2ProductOperator(SKFemBilinearFormOperator):

    def __init__(self, basis, dirichlet_dofs=None, dirichlet_clear_diag=False, coefficient_function=None,
                 name=None):
        super().__init__(basis, dirichlet_dofs=dirichlet_dofs, dirichlet_clear_diag=dirichlet_clear_diag, name=name)
        self.__auto_init(locals())

    def build_form(self, mu):
        def bf(u, v, w):
            if self.coefficient_function is None:
                return u * v
            else:
                c = _eval_pymor_function(self.coefficient_function, w.x, mu)
                return u * v * c
        return bf


class AdvectionOperator(SKFemBilinearFormOperator):

    def __init__(self, basis, advection_function, dirichlet_dofs=None, dirichlet_clear_diag=False, name=None):
        super().__init__(basis, dirichlet_dofs=dirichlet_dofs, dirichlet_clear_diag=dirichlet_clear_diag, name=name)
        self.__auto_init(locals())

    def build_form(self, mu):
        def bf(u, v, w):
            c = -_eval_pymor_function(self.advection_function, w.x, mu)
            return u * dot(c, grad(v))
        return bf


class L2Functional(SKFemLinearFormOperator):

    def __init__(self, basis, function, dirichlet_dofs=None, dirichlet_data=None, name=None):
        super().__init__(basis, dirichlet_dofs=dirichlet_dofs, name=name)
        self.__auto_init(locals())

    def build_form(self, mu):
        def lf(u, w):
            f = _eval_pymor_function(self.function, w.x, mu)
            return u * f
        return lf


class BoundaryDirichletFunctional(NumpyMatrixBasedOperator):
    sparse = False
    source = NumpyVectorSpace(1)

    def __init__(self, basis, dirichlet_data, dirichlet_dofs=None, name=None):
        assert dirichlet_data.shape_range == ()
        self.__auto_init(locals())
        self.range = NumpyVectorSpace(basis.N)

    def _assemble(self, mu=None):
        D = projection(lambda x: _eval_pymor_function(self.dirichlet_data, x, mu=mu), self.basis,
                       I=self.dirichlet_dofs)
        F = np.zeros(self.range.dim)
        F[self.dirichlet_dofs] = D
        return F.reshape((-1, 1))


class SKFemVisualizer(ImmutableObject):
    def __init__(self, space, basis):
        self.__auto_init(locals())

    def visualize(self, U, **kwargs):
        if not isinstance(U, VectorArray):
            raise NotImplementedError
        if len(U) > 1:
            raise NotImplementedError

        plot(self.basis, U.to_numpy().ravel(), colorbar=True, **kwargs)
        show()


def discretize_stationary_cg(analytical_problem, diameter=None, mesh_type=None, element=None, preassemble=True):
    """Discretizes a |StationaryProblem| with finite elements using scikit-fem.

    Parameters
    ----------
    analytical_problem
        The |StationaryProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
    mesh_type
        If not `None`, a `skfem.Mesh` to be used for discretizing the domain of
        `analytical_problem`.
    element
        If not `None`, the `skfem.Element` to be used for building the finite element space.
        If `None`, `mesh.elem()` is used.
    preassemble
        If `True`, preassemble all operators in the resulting |Model|.

    Returns
    -------
    m
        The |Model| that has been generated.
    data
        Dictionary with the following entries:

            :mesh:             The generated `skfem.Mesh`.
            :basis:            The generated `skfem.Basis`.
            :boundary_facets:  Dict of `boundary_facets` of `mesh` per boundary type.
            :dirichlet_dofs:   DOFs of the `skfem.Basis` associated with the Dirichlet boundary.
            :unassembled_m:    In case `preassemble` is `True`, the generated |Model|
                               before preassembling operators.
    """
    assert isinstance(analytical_problem, StationaryProblem)

    p = analytical_problem

    if p.nonlinear_advection is not None:
        raise NotImplementedError
    if p.nonlinear_advection_derivative is not None:
        raise NotImplementedError
    if p.nonlinear_reaction is not None:
        raise NotImplementedError
    if p.nonlinear_reaction_derivative is not None:
        raise NotImplementedError
    if not p.domain.boundary_types <= {'dirichlet', 'neumann'}:
        raise NotImplementedError

    mesh, boundary_facets = discretize_domain(p.domain, mesh_type=mesh_type, diameter=diameter)
    element = element or mesh.elem()
    basis = Basis(mesh, element)
    dirichlet_dofs = basis.get_dofs(boundary_facets['dirichlet']) if p.domain.has_dirichlet else None

    Li = [DiffusionOperator(basis, ConstantFunction(0., p.domain.dim), dirichlet_dofs=dirichlet_dofs,
                            name='boundary_part')]
    coefficients = [1.]

    _assemble_operator(
        p.diffusion, 'diffusion',
        lambda f, n: DiffusionOperator(basis, f, dirichlet_dofs=dirichlet_dofs, dirichlet_clear_diag=True, name=n),
        Li, coefficients
    )

    _assemble_operator(
        p.advection, 'advection',
        lambda f, n: AdvectionOperator(basis, f, dirichlet_dofs=dirichlet_dofs, dirichlet_clear_diag=True, name=n),
        Li, coefficients
    )

    _assemble_operator(
        p.reaction, 'reaction',
        lambda f, n: L2ProductOperator(basis, coefficient_function=f, dirichlet_dofs=dirichlet_dofs,
                                       dirichlet_clear_diag=True, name=n),
        Li, coefficients
    )

    L = LincombOperator(operators=Li, coefficients=coefficients, name='ellipticOperator')

    # right-hand side
    Fi = []
    coefficients_F = []

    _assemble_operator(
        p.rhs, 'rhs',
        lambda f, n: L2Functional(basis, f, dirichlet_dofs=dirichlet_dofs, name=n),
        Fi, coefficients_F
    )

    if p.dirichlet_data is not None and dirichlet_dofs is not None:
        dirichlet_basis = BoundaryFacetBasis(mesh, element, facets=boundary_facets['dirichlet'])
        _assemble_operator(
            p.dirichlet_data, 'dirichlet',
            lambda f, n: BoundaryDirichletFunctional(dirichlet_basis, f, dirichlet_dofs=dirichlet_dofs, name=n),
            Fi, coefficients_F
        )

    if p.neumann_data is not None and p.domain.has_neumann:
        neumann_basis = BoundaryFacetBasis(mesh, element, facets=boundary_facets['neumann'])
        _assemble_operator(
            p.neumann_data, 'neumann',
            lambda f, n: L2Functional(neumann_basis, f, name=n),
            Fi, coefficients_F, negative=True
        )

    F = LincombOperator(operators=Fi, coefficients=coefficients_F, name='rhsOperator')

    l2_product = L2ProductOperator(basis)
    h1_semi_product = DiffusionOperator(basis, ConstantFunction(1, p.domain.dim))

    products = {
        'l2': l2_product,
        'h1_semi': h1_semi_product,
        'h1': l2_product + h1_semi_product,
    }

    if p.outputs:
        if any(v[0] not in ('l2', 'l2_boundary') for v in p.outputs):
            raise NotImplementedError
        outputs = []
        boundary_basis = None
        for v in p.outputs:
            if v[0] == 'l2':
                outputs.append(
                    _assemble_operator(v[1], 'l2_output', lambda f, n: L2Functional(basis, f, name=n).H)
                )
            else:
                boundary_basis = boundary_basis or BoundaryFacetBasis(mesh, element)
                outputs.append(
                    _assemble_operator(v[1], 'l2_boundary_output',
                                       lambda f, n: L2Functional(boundary_basis, f, name=n).H)
                )
        if len(outputs) > 1:
            from pymor.operators.block import BlockColumnOperator
            from pymor.operators.constructions import NumpyConversionOperator
            output_functional = BlockColumnOperator(outputs)
            output_functional = NumpyConversionOperator(output_functional.range) @ output_functional
        else:
            output_functional = outputs[0]
    else:
        output_functional = None

    m  = StationaryModel(L, F, output_functional=output_functional, products=products,
                         visualizer=SKFemVisualizer(L.source, basis),
                         name=f'{p.name}_CG')

    data = {
        'mesh': mesh,
        'basis': basis,
        'boundary_facets': boundary_facets,
        'dirichlet_dofs': dirichlet_dofs,
    }

    if preassemble:
        data['unassembled_m'] = m
        m = preassemble_(m)

    return m, data


def _eval_pymor_function(f, x, mu):
    if len(f.shape_range) > 1:
        raise NotImplementedError
    x = np.moveaxis(x, 0, -1)
    fx = f(x, mu=mu)
    if len(f.shape_range) == 1:
        fx = np.moveaxis(fx, -1, 0)
    return fx


def _assemble_operator(function, name, factory, ops=None, coeffs=None, negative=False):
    if isinstance(function, LincombFunction):
        operators = [factory(-f if negative else f, name=f'name_{i}') for i, f in enumerate(function)]
        if ops is not None:
            ops.extend(operators)
            coeffs.extend(function.coefficients)
        else:
            return LincombOperator(operators, function.coefficients, name=name)
    elif function is not None:
        operator = factory(-function if negative else function, name)
        if ops is not None:
            ops.append(operator)
            coeffs.append(1.)
        else:
            return operator
