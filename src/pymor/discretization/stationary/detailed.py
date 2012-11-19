#!/usr/bin/env python

from __future__ import print_function, division
import pymor.core


class Interface(pymor.core.BasicInterface):

    id = 'discretization.stationary.detailed'

    def __str__(self):
        return id


class Scalar(Interface):

    id = Interface.id + '.scalar'

    def __init__(self, problem, solver):
        self.problem = problem
        self.solver = solver
