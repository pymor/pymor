from __future__ import absolute_import, division, print_function

from pymor.core import interfaces
from pymor.functions import parametric


class Interface(parametric.Interface):

    id = 'functions.parametric.separable'
    size = 0
    components = []
    coefficients = []

    @interfaces.abstractmethod
    def evaluate(self, x, mu):
        pass
