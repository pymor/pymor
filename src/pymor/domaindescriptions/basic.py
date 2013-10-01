# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.domaindescriptions import BoundaryType
from .interfaces import DomainDescriptionInterface


class RectDomain(DomainDescriptionInterface):
    '''Describes a rectangular domain.

    Boundary types can be associated edgewise.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain.
    left
        The `BoundaryType` of the left edge.
    right
        The `BoundaryType` of the right edge.
    top
        The `BoundaryType` of the top edge.
    bottom
        The `BoundaryType` of the bottom edge.

    Attributes
    ----------
    domain
    left
    right
    top
    bottom
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


class CylindricalDomain(DomainDescriptionInterface):
    '''Describes a cylindrical domain.

    Boundary types can be associated edgewise.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain. The left and right edge are identified.
    top
        The `BoundaryType` of the top edge.
    bottom
        The `BoundaryType` of the bottom edge.

    Attributes
    ----------
    domain
    top
    bottom
    '''

    def __init__(self, domain=[[0, 0], [1, 1]], top=BoundaryType('dirichlet'), bottom=BoundaryType('dirichlet')):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        self.boundary_types = set((top, bottom))
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


class TorusDomain(DomainDescriptionInterface):
    '''Describes a domain with the topology of a torus.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain. The left and right edge are identified, as well as the
        bottom and top edge

    Attributes
    ----------
    domain
    '''

    def __init__(self, domain=[[0, 0], [1, 1]]):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        self.boundary_types = set()
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


class LineDomain(DomainDescriptionInterface):
    '''Describes an interval domain.

    Boundary types can be associated edgewise.

    Parameters
    ----------
    domain
        List [x_l, x_r] providing the left and right endpoint.
    left
        The `BoundaryType` of the left endpoint.
    right
        The `BoundaryType` of the right endpoint.

    Attributes
    ----------
    domain
    left
    right
    '''

    def __init__(self, domain=[0, 1], left=BoundaryType('dirichlet'), right=BoundaryType('dirichlet')):
        assert domain[0] <= domain[1]
        self.boundary_types = set((left, right))
        self.left = left
        self.right = right
        self.domain = np.array(domain)

    @property
    def width(self):
        return self.domain[1] - self.domain[0]


class CircleDomain(DomainDescriptionInterface):
    '''Describes a domain with the topology of a circle, i.e. a line with
    identified end points.

    Parameters
    ----------
    domain
        List [x_l, x_r] providing the left and right endpoint.

    Attributes
    ----------
    domain
    '''

    def __init__(self, domain=[0, 1]):
        assert domain[0] <= domain[1]
        self.domain = np.array(domain)

    @property
    def width(self):
        return self.domain[1] - self.domain[0]
