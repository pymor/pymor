# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.domaindescriptions.interfaces import DomainDescriptionInterface, KNOWN_BOUNDARY_TYPES


class RectDomain(DomainDescriptionInterface):
    """Describes a rectangular domain.

    Boundary types can be associated edgewise.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain.
    left
        The boundary type of the left edge.
    right
        The boundary type of the right edge.
    top
        The boundary type of the top edge.
    bottom
        The boundary type of the bottom edge.

    Attributes
    ----------
    domain
    left
    right
    top
    bottom
    """

    dim = 2

    def __init__(self, domain=([0, 0], [1, 1]), left='dirichlet', right='dirichlet',
                 top='dirichlet', bottom='dirichlet'):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        for bt in (left, right, top, bottom):
            if bt is not None and bt not in KNOWN_BOUNDARY_TYPES:
                self.logger.warning(f'Unknown boundary type: {bt}')
        domain = np.array(domain)
        self.__auto_init(locals())
        self.boundary_types = frozenset({left, right, top, bottom})

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
    """Describes a cylindrical domain.

    Boundary types can be associated edgewise.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain. The left and right edge are identified.
    top
        The boundary type of the top edge.
    bottom
        The boundary type of the bottom edge.

    Attributes
    ----------
    domain
    top
    bottom
    """

    dim = 2

    def __init__(self, domain=([0, 0], [1, 1]), top='dirichlet', bottom='dirichlet'):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        for bt in (top, bottom):
            if bt is not None and bt not in KNOWN_BOUNDARY_TYPES:
                self.logger.warning(f'Unknown boundary type: {bt}')
        domain = np.array(domain)
        self.__auto_init(locals())
        self.boundary_types = frozenset({top, bottom})

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
    """Describes a domain with the topology of a torus.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain. The left and right edge are identified, as well as the
        bottom and top edge

    Attributes
    ----------
    domain
    """

    dim = 2

    def __init__(self, domain=([0, 0], [1, 1])):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        self.boundary_types = frozenset()
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
    """Describes an interval domain.

    Boundary types can be associated edgewise.

    Parameters
    ----------
    domain
        List [x_l, x_r] providing the left and right endpoint.
    left
        The boundary type of the left endpoint.
    right
        The boundary type of the right endpoint.

    Attributes
    ----------
    domain
    left
    right
    """

    dim = 1

    def __init__(self, domain=(0, 1), left='dirichlet', right='dirichlet'):
        assert domain[0] <= domain[1]
        for bt in (left, right):
            if bt is not None and bt not in KNOWN_BOUNDARY_TYPES:
                self.logger.warning(f'Unknown boundary type: {bt}')
        domain = np.array(domain)
        self.__auto_init(locals())
        self.boundary_types = frozenset({left, right})

    @property
    def width(self):
        return self.domain[1] - self.domain[0]


class CircleDomain(DomainDescriptionInterface):
    """Describes a domain with the topology of a circle, i.e. a line with
    identified end points.

    Parameters
    ----------
    domain
        List [x_l, x_r] providing the left and right endpoint.

    Attributes
    ----------
    domain
    """

    dim = 1

    def __init__(self, domain=(0, 1)):
        assert domain[0] <= domain[1]
        self.domain = np.array(domain)

    @property
    def width(self):
        return self.domain[1] - self.domain[0]
