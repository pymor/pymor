#!/usr/bin/env python

# pymor
from pymor.core import interfaces


class Interface(interfaces.BasicInterface):

    id = 'functions.parametric'
    dim_domain = (0, 0)
    dim_range = (0, 0)
    name = id

    def __str__(self):
        return ('{name} ({id}): '
                + 'R^{dim_domain_x} x R^{dim_domain_mu} -> R^{dim_range_x} x R^{dim_range_mu}'
                ).format(name=self.name,
                         dim_domain_x=self.dim_domain[0],
                         dim_domain_mu=self.dim_domain[1],
                         dim_range_x=self.dim_range[0],
                         dim_range_mu=self.dim_range[1],
                         id=self.id)

    @interfaces.abstractmethod
    def evaluate(self, x, mu):
        pass
