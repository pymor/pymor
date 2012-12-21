#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import numpy as np
import scipy.sparse as sparse

from pymor.core import interfaces
from pymor.core.exceptions import warn, CallOrderWarning
import pymor.problem.stationary.linear.elliptic.analytical as analyticalproblem
import pymor.grid.oned as grid
import pymor.common.boundaryinfo.oned as boundaryinfo
import pymor.discreteoperators.stationary.linear as operator
import pymor.common.discretefunctional.linear as functional
import pymor.problem.stationary.linear.discrete as discreteproblem
import pymor.solver.stationary.linear.scalar as solver
import pymor.discretizations.stationary.detailed as discretization
import pymor.common.discretefunction.continuous as discretefunction


class Interface(interfaces.BasicInterface):

    id = 'discretizer.stationary.linear.elliptic.cg'
    trial_order = -1
    test_order = -1
    data_order = -1

    def __str__(self):
        return id

    @interfaces.abstractmethod
    def discretize(self):
        pass

    @interfaces.abstractmethod
    def discretization(self):
        pass


class P1(Interface):

    id = Interface.id + '.p1'
    trial_order = 1
    test_order = 1
    data_order = 0

    def __init__(self, problem=analyticalproblem.Default(),
                 grid=grid.Oned(),
                 boundaryinfo=boundaryinfo.AllDirichletBoundaryInfo()):
        self.problem = problem
        self.grid = grid
        self.boundaryinfo = boundaryinfo
        assert self.grid.dim == 1
        assert self.grid.volumes(0).min() == self.grid.volumes(0).max()
        assert self.problem.diffusion().dim_domain == self.grid.dim
        assert self.problem.diffusion().dim_range == 1
        self._discretized = False

    def discretize(self):
        if not self._discretized:
            # create common stuff for operator and functional
            self._h = self.grid.volumes(0).min()
            self._n = self.grid.size(0)
            # project data functions only once
            self._diffusion = self.problem.diffusion().evaluate(self.grid.centers(1))
            self._force = self.problem.force().evaluate(self.grid.centers(1))
            # create some maps
            self._zero_to_n_minus_two = range(self._n - 1)
            self._one_to_n_minus_one = range(1, self._n)
            self._two_to_n = range(2, self._n + 1)
            # assemble
            self._assemble_operator()
            self._assemble_functional()
            # clear common stuff
            del self._diffusion
            del self._force
            del self._zero_to_n_minus_two
            del self._one_to_n_minus_one
            del self._two_to_n
            # finished
            self._discretized = True
        # create discrete problem
        discrete_problem = discreteproblem.Scalar(self.grid,
                                                  self.problem,
                                                  self.boundaryinfo,
                                                  self._operator,
                                                  self._functional)
        # create discrete solver
        discrete_solver = solver.Scipy()
        # create detailed discretization
        self._discretization = discretization.Scalar(discrete_problem,
                                                     discrete_solver,
                                                     self)
        return self._discretization

    def discretization(self):
        if not self._discretized:
            warn('Please call \'discretize()\' before calling \'discretization(),'
                 + 'calling \'discretize()\' now!\'',
                 CallOrderWarning)
            self.discretize()
        return self._discretization

    def create_discrete_function(self, vector, name):
        return discretefunction.P1(self.grid, vector, name)

    def _assemble_operator(self):
        # get stuff
        h = self._h
        n = self._n
        diffusion = self._diffusion
        # get maps
        zero_to_n_minus_two = self._zero_to_n_minus_two
        one_to_n_minus_one = self._one_to_n_minus_one
        two_to_n = self._two_to_n
        # assemble diagonal entries
        rows_1 = one_to_n_minus_one
        cols_1 = one_to_n_minus_one
        vals_1 = ((1.0 / h) * diffusion[zero_to_n_minus_two]
                        + (1.0 / h) * diffusion[one_to_n_minus_one])
        # assemble lower diagonal entries
        rows_2 = one_to_n_minus_one
        cols_2 = zero_to_n_minus_two
        vals_2 = (-1.0 / h) * diffusion[zero_to_n_minus_two]
        # assemble upper diagonal entries
        rows_3 = one_to_n_minus_one
        cols_3 = two_to_n
        vals_3 = (-1.0 / h) * diffusion[one_to_n_minus_one]
        # assemble left boundary values
        rows_4 = [0, 0]
        cols_4 = [0, 1]
        vals_4 = np.zeros(2)
        if self.boundaryinfo.left() == 'dirichlet':
            vals_4[0] = 1.0
        elif self.boundaryinfo.left() == 'neumann':
            vals_4[0] = (1.0 / h) * diffusion[0]
            vals_4[1] = (-1.0 / h) * diffusion[0]
        else:
            raise ValueError('wrong boundary type!')
        # assemble right boundary values
        rows_5 = [n, n]
        cols_5 = [n, n - 1]
        vals_5 = np.zeros(2)
        if self.boundaryinfo.right() == 'dirichlet':
            vals_5[0] = 1.0
        elif self.boundaryinfo.right() == 'neumann':
            vals_5[0] = (1.0 / h) * diffusion[n - 1]
            vals_5[1] = (-1.0 / h) * diffusion[n - 1]
        else:
            raise ValueError('wrong boundary type!')
        # create sparse matrix
        rows = np.array(rows_1 + rows_2 + rows_3 + rows_4 + rows_5)
        cols = np.array(cols_1 + cols_2 + cols_3 + cols_4 + cols_5)
        vals = np.concatenate((vals_1, vals_2, vals_3, vals_4, vals_5), axis=0)
        matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
        # create operator
        self._operator = operator.ScipySparse(matrix)

    def _assemble_functional(self):
        # get stuff
        h = self._h
        n = self._n
        diffusion = self._diffusion
        force = self._force
        # get maps
        zero_to_n_minus_two = self._zero_to_n_minus_two
        one_to_n_minus_one = self._one_to_n_minus_one
        # assemble dense vector
        vector = np.zeros(n + 1)
        vector[one_to_n_minus_one] = (0.5 * h) * (force[zero_to_n_minus_two]
                                                  + force[one_to_n_minus_one]);
        # assemble left boundary values
        x_left = self.grid.centers(1)[0]
        if self.boundaryinfo.left() == 'dirichlet':
            vector[0] = self.problem.dirichlet().evaluate(x_left)
        elif self.boundaryinfo.left() == 'neumann':
            vector[0] = (h * force[0]
                         + diffusion[0] * self.problem.neumann.evaluate(x_left));
        else:
            raise ValueError('wrong boundary type!')
        # assemble right boundary values
        x_right = self.grid.centers(1)[-1]
        if self.boundaryinfo.right() == 'dirichlet':
            vector[-1] = self.problem.dirichlet().evaluate(x_right)
        elif self.boundaryinfo.right() == 'neumann':
            vector[-1] = (h * force[-1]
                         + diffusion[-1] * self.problem.neumann.evaluate(x_right));
        else:
            raise ValueError('wrong boundary type!')
        # create functional
        self._functional = functional.NumpyDense(vector)


if __name__ == '__main__':
    print('creating analytical problem... ', end='')
    problem = analyticalproblem.Default()
    print('done (' + problem.id + ')')
    print('creating grid... ', end='')
    grid = grid.Oned([0., 1.], 1000)
    print('done ({name}, ' 'size {size})'.format(name=grid.id, size=grid.size()))
    print('creating boundaryinfo... ', end='')
    boundaryinfo = boundaryinfo.AllDirichletBoundaryInfo()
    print('done (' + boundaryinfo.id + ')')
    print('creating discretizer... ', end='')
    discretizer = P1(problem, grid)
    print('done (' + discretizer.id + ')')
    print('creating discretization... ', end='')
    discretization = discretizer.discretize()
    print('done (' + discretization.id + ')')
    print('solving... ', end='')
    solution = discretization.solve()
    solution.name = discretizer.id + '.solution'
    print('done')
    discretization.visualize(solution)
