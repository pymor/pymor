
from .interfaces import Interface
from pymor.common import function

def Default(Interface):

    def __init__(self, name='problem.stationary.linear.alliptic.default'):
        self._name = name
        self._diffusion = function.Constant(1)
        self._force = function.Constant(1)
        self._dirichlet = function.Constant(0)
        self._neumann = function.Constant(0)

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
