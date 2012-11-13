
from __future__ import print_function

import abc

from pymor import core
from pymor.common import function


class Interface(core.BaseInterface):

    def name(self):
        return 'problem.stationary.linear.elliptic'

    @abc.abstractmethod
    def diffusion(self):
        pass

    @abc.abstractmethod
    def force(self):
        pass

    @abc.abstractmethod
    def dirichlet(self):
        pass

    @abc.abstractmethod
    def neumann(self):
        pass


class Default(Interface):

    def __init__(self, name='problem.stationary.linear.elliptic.default'):
        self._name = name
        self._diffusion = function.Constant(1, 'diffusion')
        self._force = function.Constant(1, 'force')
        self._dirichlet = function.Constant(0, 'dirichlet')
        self._neumann = function.Constant(0, 'neumann')

    def name(self):
        return self._name

    def diffusion(self):
        return self._diffusion

    def force(self):
        return self._force

    def dirichlet(self):
        return self._dirichlet

    def neumann(self):
        return self._neumann


if __name__ == "__main__":
    print('creating problem... ', end='')
    p = Default()
    print('done (' + p.name() + ')')