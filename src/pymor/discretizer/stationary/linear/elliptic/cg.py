#!/usr/bin/env python

from __future__ import print_function, division
import warnings
import numpy as np
import scipy.sparse
import pymor.core
import pymor.problem.stationary.linear.elliptic.analytical
import pymor.common.boundaryinfo
import pymor.grid.oned
import pymor.common.discreteoperator.stationary.linear
from pymor.core.warnings import CallOrderWarning

class Interface(pymor.core.BasicInterface):

    id = 'discretizer.stationary.linear.elliptic.cg'
    trial_order = -1
    test_order = -1
    data_order = -1

    def __str__(self):
        return id


class P1(Interface):

    id = Interface.id + '.p1'
    trial_order = 1
    test_order = 1
    data_order = 0

    def __init__(self, problem=pymor.problem.stationary.linear.elliptic.analytical.Default(),
                 grid=pymor.grid.oned.Oned(),
                 boundaryinfo=pymor.common.boundaryinfo.oned.AllDirichlet()):
        self.problem = problem
        self.grid = grid
        self.boundaryinfo = boundaryinfo
        assert self.grid.dim == 1
        assert self.grid.volumes(0).min() == self.grid.volumes(0).max()
        assert self.problem.diffusion().dim_domain == self.grid.dim
        assert self.problem.diffusion().dim_range == 1
        self._initialized = False
        self._assembled_operator = False
        self._assembled_functional = False

    def operator(self):
        if not self._assembled_operator:
            warnings.warn('Please call init() before calling operator()! Calling init() now...', CallOrderWarning)
            self._assemble_operator()
        return self._operator

    def functional(self):
        if not self._assembled_functional:
            warnings.warn('Please call init() before calling functional()! Calling init() now...', CallOrderWarning)
            self._assemble_functional()()

    def init(self):
        if not self._initialized:
            self._assemble_operator()
            self._assemble_functional()
            self._initialized = True

    def _assemble_operator(self):
        if not self._assembled_operator:
            # preparations
            h = self.grid.volumes(0).min()
            n = self.grid.size(0)
            # project datafunctions
            diffusion = self.problem.diffusion().evaluate(self.grid.centers(1))
            # some maps
            zero_to_n_minus_two = range(n - 1)
            one_to_n_minus_one = range(1, n)
            two_to_n = range(2, n + 1)
            # assemble diagonal entries
            rows_1 = one_to_n_minus_one
            cols_1 = one_to_n_minus_one
            vals_1 = ((1.0/h) * diffusion[zero_to_n_minus_two]
                            + (1.0/h) * diffusion[one_to_n_minus_one])
            # assemble lower diagonal entries
            rows_2 = one_to_n_minus_one
            cols_2 = zero_to_n_minus_two
            vals_2 = (-1.0/h) * diffusion[zero_to_n_minus_two]
            # assemble upper diagonal entries
            rows_3 = one_to_n_minus_one
            cols_3 = two_to_n
            vals_3 = (-1.0/h) * diffusion[one_to_n_minus_one]
            # assemble left boundary values
            rows_4 = [0, 0]
            cols_4 = [0, 1]
            vals_4 = np.zeros(2)
            if self.boundaryinfo.left() == 'dirichlet':
                vals_4[0] = 1.0
            elif self.boundaryinfo.left() == 'neumann':
                vals_4[0] = (1.0/h) * diffusion[0]
                vals_4[1] = (-1.0/h) * diffusion[0]
            else:
                raise ValueError('wrong boundary type!')
            # assemble left boundary values
            rows_5 = [n, n]
            cols_5 = [n, n - 1]
            vals_5 = np.zeros(2)
            if self.boundaryinfo.right() == 'dirichlet':
                vals_5[0] = 1.0
            elif self.boundaryinfo.right() == 'neumann':
                vals_5[0] = (1.0/h) * diffusion[n - 1]
                vals_5[1] = (-1.0/h) * diffusion[n - 1]
            else:
                raise ValueError('wrong boundary type!')
            # create sparse matrix
            rows = np.array(rows_1 + rows_2 + rows_3 + rows_4 + rows_5)
            cols = np.array(cols_1 + cols_2 + cols_3 + cols_4 + cols_5)
            vals = np.concatenate((vals_1, vals_2, vals_3, vals_4, vals_5), axis=0)
            matrix = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n + 1,n + 1))
            # create operator
            self._operator = pymor.common.discreteoperator.stationary.linear.ScipySparse(matrix)
            # finished
            self._assembled_operator = True


    def _assemble_functional(self):
        if not self._assembled_functional:
            # finished
            self._assembled_functional = True


if __name__ == '__main__':
    print('creating problem... ', end='')
    problem = pymor.problem.stationary.linear.elliptic.analytical.Default()
    print('done (' + problem.id + ')')
    print('creating grid... ', end='')
    grid = pymor.grid.oned.Oned()
    print('done ({name}, ' 'size {size})'.format(name=grid.id, size=grid.size()))
    print('creating boundaryinfo... ', end='')
    boundaryinfo = pymor.common.boundaryinfo.oned.AllDirichlet()
    print('done (' + boundaryinfo.id + ')')
    print('creating discretizer (', end='')
    discretizer = P1(problem, grid)
    print(discretizer.id + '):')
    print('  initializing... ', end='')
    discretizer.init()
    print('done')