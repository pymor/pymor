#!/usr/bin/env python

# only for name == main
from __future__ import print_function

# pymor
from pymor.core import interfaces


class Interface(interfaces.BasicInterface):

    id = 'common.boundaryinfo.oned'

    def __init__(self, left='dirichlet', right='dirichlet'):
        assert (left == 'dirichlet' or left == 'neumann')
        assert (right == 'dirichlet' or right == 'neumann')
        self._left = left
        self._right = right

    def __str__(self):
        return id

    def left(self):
        return self._left

    def right(self):
        return self._right


class AllDirichletBoundaryInfo(Interface):

    id = Interface.id + '.alldirichlet'


class AllNeumann(Interface):

    id = Interface.id + '.allneumann'

    def __init__(self):
        Interface.__init__(self, 'neumann', 'neumann')


if __name__ == '__main__':
    print('testing ', end='')
    dirichlet = AllDirichletBoundaryInfo()
    print('{id}... '.format(id=dirichlet.id), end='')
    assert dirichlet.left() == 'dirichlet'
    assert dirichlet.right() == 'dirichlet'
    print('done')
    print('testing ', end='')
    neumann = AllNeumann()
    print('{id}... '.format(id=neumann.id), end='')
    assert neumann.left() == 'neumann'
    assert neumann.right() == 'neumann'
    print('done')
