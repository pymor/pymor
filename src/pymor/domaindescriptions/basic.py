from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pymor.core as core
from pymor.domaindescriptions import BoundaryType
from .interfaces import DomainDescriptionInterface


class RectDomain(DomainDescriptionInterface):
    '''Describes a rectangular domain. Different boundary types can be associated
    to each side.
    '''

    def __init__(self, domain=[[0, 0], [1, 1]], left=BoundaryType('dirichlet'), right=BoundaryType('dirichlet'),
                 top=BoundaryType('dirichlet'), bottom=BoundaryType('dirichlet')):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        self.boundary_types = set((left, right, top, bottom))
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.domain = np.array(domain)

    @property
    def lower_left(self):
        return self.domain[0]

    @property
    def upper_right(self):
        return self.domain[1]

    @property
    def width(self):
        return self.domain[1, 0] - self.domain[0, 0]

    @property
    def height(self):
        return self.domain[1, 1] - self.domain[0, 1]

    @property
    def volume(self):
        return self.width * self.height

    @property
    def diameter(self):
        return np.sqrt(self.width ** 2 + self.height ** 2)
