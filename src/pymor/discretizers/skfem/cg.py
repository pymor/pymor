# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module provides some operators for continuous finite element discretizations."""

import numpy as np
from skfem import Basis, BilinearForm, LinearForm, asm, enforce
from skfem.helpers import grad, dot

from pymor.algorithms.preassemble import preassemble as preassemble_
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction
from pymor.discretizers.skfem.domaindiscretizer import discretize_domain
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class SKFemBilinearFormOperator(NumpyMatrixBasedOperator):

    sparse = True

    def __init__(self, basis, dirichlet_dofs=None, dirichlet_clear_diag=False, name=None):
        self.source = self.range = NumpyVectorSpace(basis.N)
        self.__auto_init(locals())

    def build_form(mu):
        pass

    def _assemble(self, mu):
        form = self.build_form(mu)
        A = asm(form, self.basis)
        if self.dirichlet_dofs is not None:
            A.setdiag(A.diagonal())
            enforce(A, D=self.dirichlet_dofs, diag=0. if self.dirichlet_clear_diag else 1., overwrite=True)
        return A


class SKFemLinearFormOperator(NumpyMatrixBasedOperator):

    sparse = True

    def __init__(self, basis, dirichlet_dofs=None, dirichlet_data=None, name=None):
        self.source = NumpyVectorSpace(1)
        self.range = NumpyVectorSpace(basis.N)
        self.__auto_init(locals())

    def _assemble(self, mu):
        form = self.build_form(mu)
        F = asm(form, self.basis)
        if self.dirichlet_dofs is not None:
            F[self.dirichlet_dofs] = self.dirichlet_data if self.dirichlet_data is not None else 0
        return F.reshape((-1, 1))


class DiffusionOperator(SKFemBilinearFormOperator):

    def __init__(self, basis, diffusion_function, dirichlet_dofs=None, dirichlet_clear_diag=False, name=None):
        super().__init__(basis, dirichlet_dofs=dirichlet_dofs, dirichlet_clear_diag=dirichlet_clear_diag, name=name)
        self.__auto_init(locals())

    def build_form(self, mu):
        @BilinearForm
        def bf(u, v, w):
            x = np.moveaxis(w['x'], 0, -1)
            d = self.diffusion_function(x, mu=mu)
            return dot(grad(u), grad(v)) * d
        return bf


class L2Functional(SKFemLinearFormOperator):

    def __init__(self, basis, function, dirichlet_dofs=None, dirichlet_data=None, name=None):
        super().__init__(basis, dirichlet_dofs=dirichlet_dofs, dirichlet_data=dirichlet_data, name=name)
        self.__auto_init(locals())

    def build_form(self, mu):
        @LinearForm
        def lf(u, w):
            x = np.moveaxis(w['x'], 0, -1)
            f = self.function(x, mu=mu)
            return u * f
        return lf


def discretize_stationary_cg(analytical_problem, diameter=None, preassemble=True):
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
    if p.domain.boundary_types != {'dirichlet'}:
        raise NotImplementedError
    if p.dirichlet_data is not None:
        raise NotImplementedError
    if p.neumann_data is not None:
        raise NotImplementedError
    if p.outputs is not None:
        raise NotImplementedError

    mesh, boundary_facets = discretize_domain(p.domain, diameter=diameter)
    element = mesh.elem()
    basis = Basis(mesh, element)
    dirichlet_dofs = basis.get_dofs(boundary_facets['dirichlet']) if p.domain.has_dirichlet else None

    Li = [DiffusionOperator(basis, ConstantFunction(0., p.domain.dim), dirichlet_dofs=dirichlet_dofs,
                            name='boundary_part')]
    coefficients = [1.]

    # diffusion part
    if isinstance(p.diffusion, LincombFunction):
        Li += [DiffusionOperator(basis, df, dirichlet_dofs=dirichlet_dofs, dirichlet_clear_diag=True,
                                 name=f'diffusion_{i}')
               for i, df in enumerate(p.diffusion.functions)]
        coefficients += list(p.diffusion.coefficients)
    elif p.diffusion is not None:
        Li += [DiffusionOperator(basis, p.diffusion, dirichlet_dofs=dirichlet_dofs, dirichlet_clear_diag=True,
                                 name='diffusion')]
        coefficients.append(1.)

    L = LincombOperator(operators=Li, coefficients=coefficients, name='ellipticOperator')

    # right-hand side
    rhs = p.rhs or ConstantFunction(0., dim_domain=p.domain.dim)
    Fi = []
    coefficients_F = []
    if isinstance(p.rhs, LincombFunction):
        Fi += [L2Functional(basis, rh, dirichlet_dofs=dirichlet_dofs, name=f'rhs_{i}')
               for i, rh in enumerate(p.rhs.functions)]
        coefficients_F += list(p.rhs.coefficients)
    else:
        Fi += [L2Functional(basis, rhs, dirichlet_dofs=dirichlet_dofs, name='rhs')]
        coefficients_F.append(1.)

    F = LincombOperator(operators=Fi, coefficients=coefficients_F, name='rhsOperator')

    visualizer = None

    products = {}

    output_functional = None

    m  = StationaryModel(L, F, output_functional=output_functional, products=products, visualizer=visualizer,
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
