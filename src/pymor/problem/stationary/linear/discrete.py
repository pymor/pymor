#!/usr/bin/env python

from __future__ import print_function, division
import pymor.core


class Interface(pymor.core.interfaces.BasicInterface):

    id = 'problem.stationary.linear.discrete'

    def __str__(self):
        return id


class Scalar(Interface):

    id = Interface.id + '.scalar'

    def __init__(self,
                 grid,
                 analytical_problem,
                 boundaryinfo,
                 operator,
                 functional):
        self.grid = grid
        self.analytical_problem = analytical_problem
        self.boundaryinfo = boundaryinfo
        self.operator = operator
        self.functional = functional
