#!/usr/bin/env python

from __future__ import print_function, division
import pymor.core
import scipy.sparse


class Interface(pymor.core.BasicInterface):
    
    id = 'common.discreteoperator.stationary.linear'
    
    def apply(self):
        pass
    
    def __str__(self):
        return self.id

class ScipySparse(Interface):
    
    id = Interface.id + '.scipysparse'
    
    def __init__(self, matrix):
        '''
        here should be a contract to check if matrix is a scipy.sparse
        '''
        self.matrix = matrix
        self.size = self.matrix.shape