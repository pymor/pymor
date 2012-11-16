#!/usr/bin/env python

from __future__ import print_function
#import numpy as np
from pymor import core
#from pymor.core.decorators import contract
import pymor.problem.stationary.linear.elliptic.analytical
import pymor.grid.oned


class DiscontinuousGalerkin(core.BasicInterface):

#    @contract
    def __init__(self, problem, grid):
        '''
        :type problem: ProblemInterface
        '''
        self._name = 'discretizer.stationary.linear.elliptic.fem'
        self._problem = problem
        self._grid = grid
        self._assembled = False

    def problem(self):
        return self._problem

    def name(self):
        return self._name

    def init(self):
        if not self._assembled:
            # assemble
            self._assembled = True


if __name__ == '__main__':
    print('creating problem...', end='')
    problem = pymor.problem.stationary.linear.elliptic.analytical.Default()
    print('done (' + problem.name() + ')')
    print('creating grid...', end='')
    grid = pymor.grid.oned.Oned()
    print('done ({name}, ' 'size {size})'.format(name=grid.name(), size=grid.size()))
    print('creating discretizer (', end='')
    discretizer = DiscontinuousGalerkin(problem, grid)
    print(discretizer.name() + '):')
    print('  initializing...', end='')
    discretizer.init()
    print('done')