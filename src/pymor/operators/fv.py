# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from pymor.la import NumpyVectorArray
from pymor.grids.referenceelements import triangle, line
from pymor.operators.interfaces import OperatorInterface, LinearOperatorInterface
from pymor.operators.numpy import NumpyLinearOperator
from pymor.tools.inplace import iadd_masked, isub_masked


class NonlinearAdvectionLaxFriedrichs(OperatorInterface):
    '''Nonlinear Finite Volume Advection operator using Lax-Friedrichs-Flux.

    Currently we assume Dirichlet-Zero on the whole boundary.
    '''

    type_source = type_range = NumpyVectorArray

    def __init__(self, grid, flux, lmbda=1.0, name=None):
        super(NonlinearAdvectionLaxFriedrichs, self).__init__()
        self.grid = grid
        self.flux = flux
        self.lmbda = lmbda
        self.name = name
        self.build_parameter_type(inherits={'flux': flux})
        self.dim_source = self.dim_range = grid.size(0)


    def apply(self, U, ind=None, mu=None):
        assert isinstance(U, NumpyVectorArray)
        assert U.dim == self.dim_source

        ind = xrange(len(U)) if ind is None else ind
        U = U._array
        R = np.zeros((len(ind), self.dim_source))

        grid = self.grid
        N = grid.neighbours(0, 0)
        SUPE = grid.superentities(1, 0)
        SUPI = grid.superentity_indices(1, 0)
        assert SUPE.ndim == 2
        VOLS = grid.volumes(1)

        for i, j in enumerate(ind):
            Ui = U[j]
            Ri = R[i]

            F = self.flux(Ui, self.map_parameter(mu, 'flux'))
            F_edge = F[SUPE]
            F_edge[SUPE == -1] = 0
            F_edge = np.sum(np.sum(F_edge, axis=1) * grid.unit_outer_normals()[SUPE[:,0], SUPI[:,0]], axis=1)

            U_edge = Ui[SUPE]
            U_edge[SUPE == -1] = 0
            U_edge = (U_edge[:,0] - U_edge[:,1]) * (1. / self.lmbda)

            TOT_edge = F_edge + U_edge
            TOT_edge *= 0.5 * VOLS

            # for k in xrange(len(TOT_edge)):
            #     Ri[SUPE[k,0]] += TOT_edge[k]
            #     Ri[SUPE[k,1]] -= TOT_edge[k]
            # Ri[SUPE[:,0]] += TOT_edge
            # Ri[SUPE[:,1]] -= TOT_edge
            iadd_masked(Ri, TOT_edge, SUPE[:,0])
            isub_masked(Ri, TOT_edge, SUPE[:,1])

        R /= grid.volumes(0)

        return NumpyVectorArray(R)


class L2Product(LinearOperatorInterface):
    '''Operator representing the L2-product for finite volume functions.

    To evaluate the product use the apply2 method.

    Parameters
    ----------
    grid
        The grid on which to assemble the product.
    name
        The name of the product.
    '''

    type_source = type_range = NumpyVectorArray
    sparse = True

    def __init__(self, grid, name=None):
        super(L2Product, self).__init__()
        self.dim_source = grid.size(0)
        self.dim_range = self.dim_source
        self.grid = grid
        self.name = name

    def _assemble(self, mu=None):
        assert mu is None

        A = diags(self.grid.volumes(0), 0)

        return NumpyLinearOperator(A)
