#!/usr/bin/env python

from __future__ import print_function, division
import pymor.core


class Interface(pymor.core.interfaces.BasicInterface):

    id = 'solver.stationary.linear.scalar'

    def __str__(self):
        return id


class Scipy(Interface):

    id = Interface.id + 'scipy'

    def __init__(self):
        pass
