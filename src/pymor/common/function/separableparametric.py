#!/usr/bin/env python

# pymor
from pymor.core import interfaces
# local
import parametric


class Interface(parametric.Interface):

    id = 'common.function.parametric.separable'
    size = 0
    components = []
    coefficients = []

    @interfaces.abstractmethod
    def evaluate(self, x, mu):
        pass
