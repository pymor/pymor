#!/usr/bin/env python

# only for name == main
from __future__ import print_function

# pymor
from pymor.core import interfaces
from pymor.common import function


class Interface(interfaces.BasicInterface):

    id = 'problem.stationary.linear.elliptic'

    def __str__(self):
        return ('{id}\n' +
                '  diffusion: {diffusion}\n' +
                '  force:     {force}\n' +
                '  dirichlet: {dirichlet}\n' +
                '  neumann:   {neumann}'
                ).format(id=self.id,
                         diffusion=self.diffusion(),
                         force=self.force(),
                         dirichlet=self.dirichlet(),
                         neumann=self.neumann())

    @interfaces.abstractmethod
    def diffusion(self):
        pass

    @interfaces.abstractmethod
    def force(self):
        pass

    @interfaces.abstractmethod
    def dirichlet(self):
        pass

    @interfaces.abstractmethod
    def neumann(self):
        pass


class Default(Interface):

    id = Interface.id + '.default'

    def __init__(self):
        self._diffusion = function.nonparametric.Constant(1, name='diffusion')
        self._force = function.nonparametric.Constant(1, name='force')
        self._dirichlet = function.nonparametric.Constant(0, name='dirichlet')
        self._neumann = function.nonparametric.Constant(0, name='neumann')

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
    print('done:')
    print(p)
