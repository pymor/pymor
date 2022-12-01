# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('FENICS')


import dolfin as df

from pymor.algorithms.preassemble import preassemble as preassemble_
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import LincombFunction
from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixBasedOperator, FenicsVisualizer
from pymor.discretizers.fenics.domaindiscretizer import discretize_domain
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator, NumpyConversionOperator
from pymor.operators.block import BlockColumnOperator


def discretize_stationary_cg(analytical_problem, diameter=None, degree=1, preassemble=True):
    """Discretizes a |StationaryProblem| with finite elements using FEniCS.

    Parameters
    ----------
    analytical_problem
        The |StationaryProblem| to discretize.
    diameter
        If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
    degree
        polynomial degree of the finite element.
    preassemble
        If `True`, preassemble all operators in the resulting |Model|.

    Returns
    -------
    m
        The |Model| that has been generated.
    data
        Dictionary with the following entries:

            :mesh:             The generated dolfin mesh object.
            :boundary_mask:    Codim-1 `MeshFunctionSizet` indicating which boundary type a
                               bundary facet belongs to.
            :boundary_ids:     Dict mapping boundary types to ids used in `boundary_mask`.
            :unassembled_m:    In case `preassemble` is `True`, the generated |Model|
                               before preassembling operators.
    """
    assert isinstance(analytical_problem, StationaryProblem)

    p = analytical_problem

    if p.diffusion is not None and not p.diffusion.shape_range == ():
        raise NotImplementedError
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
    if p.dirichlet_data is not None and p.dirichlet_data.parametric:
        raise NotImplementedError

    mesh, (boundary_mask, boundary_ids) = discretize_domain(p.domain, diameter=diameter)

    V = df.FunctionSpace(mesh, 'Lagrange', degree)
    bc = df.DirichletBC(V, 0. if p.dirichlet_data is None else p.dirichlet_data.to_fenics(mesh)[0].item(),
                        boundary_mask, boundary_ids['dirichlet'])
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    dx, ds = df.dx, df.ds

    Li = [FenicsMatrixBasedOperator(df.Constant(0.)*u*v*dx, {}, bc, bc_zero=False, name='boundary_part')]
    coefficients = [1.]

    _assemble_operator(p.diffusion, lambda c: df.inner(c.item() * df.grad(u), df.grad(v)) * dx,
                       mesh, bc, True, 'diffusion',
                       Li, coefficients)

    _assemble_operator(
        p.advection, lambda c: u * sum(ci * gi for ci, gi in zip(c, df.grad(v))) * dx,
        mesh, bc, True, 'advection',
        Li, coefficients
    )

    _assemble_operator(
        p.reaction, lambda c: c * u * v * dx,
        mesh, bc, True, 'reaction',
        Li, coefficients,
    )

    L = LincombOperator(operators=Li, coefficients=coefficients, name='ellipticOperator')

    # right-hand side
    Fi = []
    coefficients_F = []

    _assemble_operator(p.rhs, lambda c: c.item() * v * dx,
                       mesh, bc, False, 'rhs',
                       Fi, coefficients_F)

    if p.neumann_data is not None and p.domain.has_neumann:
        _assemble_operator(
            p.neumann_data, lambda c: c.item() * v * ds,
            mesh, bc, False, 'neumann',
            Fi, coefficients_F, negative=True
        )

    F = LincombOperator(operators=Fi, coefficients=coefficients_F, name='rhsOperator')

    h1_0_semi_product = FenicsMatrixBasedOperator(df.inner(df.grad(u), df.grad(v))*dx, {}, bc, bc_zero=False,
                                                  name='h1_0_semi')
    l2_product = FenicsMatrixBasedOperator(u*v*dx, {}, name='l2')
    h1_semi_product = FenicsMatrixBasedOperator(df.inner(df.grad(u), df.grad(v))*dx, {}, bc, bc_zero=False,
                                                name='h1_0_semi')
    h1_product = l2_product + h1_semi_product

    products = {
        'l2': l2_product,
        'h1_semi': h1_0_semi_product,
        'h1': h1_product,
        'h1_0_semi': h1_0_semi_product,
    }

    if p.outputs:
        if any(o[0] not in ('l2', 'l2_boundary') for o in p.outputs):
            raise NotImplementedError
        outputs = []
        for o in p.outputs:
            if o[0] == 'l2':
                outputs.append(
                    _assemble_operator(o[1], lambda c: c * v * dx, mesh,
                                       functional=True, name='l2_output')
                )
            else:
                outputs.append(
                    _assemble_operator(o[1], lambda c: c * v * ds, mesh,
                                       functional=True, name='l2_boundary_output')
                )
        if len(outputs) > 1:
            output_functional = BlockColumnOperator(outputs)
            output_functional = NumpyConversionOperator(output_functional.range) @ output_functional
        else:
            output_functional = outputs[0]
    else:
        output_functional = None

    m = StationaryModel(L, F, output_functional=output_functional, products=products,
                        visualizer=FenicsVisualizer(FenicsVectorSpace(V)),
                        name=f'{p.name}_CG')

    data = {
        'mesh': mesh,
        'boundary_mask': boundary_mask,
        'boundary_ids': boundary_ids,
        'bc': bc,
    }

    if preassemble:
        data['unassembled_m'] = m
        m = preassemble_(m)

    return m, data


def _assemble_operator(function, factory,
                       mesh, bc=None, bc_zero=None, name=None,
                       ops=None, coeffs=None,
                       negative=False, functional=False):

    def assemble_op(f, name):
        coeff, params = f.to_fenics(mesh)
        return FenicsMatrixBasedOperator(factory(coeff), params,
                                         bc=bc, bc_zero=bc_zero, functional=functional, name=name)

    if isinstance(function, LincombFunction):
        operators = [assemble_op(f, f'{name}_{i}') for i, f in enumerate(function.functions)]
        cfs = [-c if negative else c for c in function.coefficients]
        if ops is not None:
            ops.extend(operators)
            coeffs.extend(cfs)
        else:
            return LincombOperator(operators, cfs, name=name)
    elif function is not None:
        operator = assemble_op(function, name)
        if ops is not None:
            ops.append(operator)
            coeffs.append(-1 if negative else 1.)
        else:
            return -operator if negative else operator
