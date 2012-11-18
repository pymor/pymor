#!/usr/bin/env python

from pymor import core

class Interface(core.BasicInterface):
    
    id = 'common.boundaryinfo'
    
    def __init__(self, left='dirichlet', right='dirichlet'):
        assert (left == 'dirichlet' or left == 'neumann')
        assert (right == 'dirichlet' or right == 'neumann')
        self._left = left
        self._right = right
    
    def __str__(self):
        return 'implement me'
    
    def left(self):
        return self._left
    
    def right(self):
        return self._right


class AllDirichlet(Interface):
    
    id = Interface.id + '.alldirichlet'