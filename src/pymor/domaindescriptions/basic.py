# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
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
        self.boundary_types = frozenset({left, right, top, bottom})
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

    def __repr__(self):
        left = ', left=' + repr(self.left) if self.left != 'dirichlet' else ''
        right = ', right=' + repr(self.right) if self.right != 'dirichlet' else ''
        top = ', top=' + repr(self.top) if self.top != 'dirichlet' else ''
        bottom = ', bottom=' + repr(self.bottom) if self.bottom != 'dirichlet' else ''
        return 'RectDomain({}{})'.format(str(self.domain).replace('\n', ','), left + right + top + bottom)


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
        self.boundary_types = frozenset({top, bottom})
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

    def __repr__(self):
        top = ', top=' + repr(self.top) if self.top != 'dirichlet' else ''
        bottom = ', bottom=' + repr(self.bottom) if self.bottom != 'dirichlet' else ''
        return 'CylindricalDomain({}{})'.format(str(self.domain).replace('\n', ','), top + bottom)


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

    def __repr__(self):
        return f'TorusDomain({str(self.domain).replace("\n", ",")})'


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
        self.boundary_types = frozenset({left, right})
        self.left = left
        self.right = right
        self.domain = np.array(domain)

    @property
    def width(self):
        return self.domain[1] - self.domain[0]

    def __repr__(self):
        left = ', left=' + repr(self.left) if self.left != 'dirichlet' else ''
        right = ', right=' + repr(self.right) if self.right != 'dirichlet' else ''
        return 'LineDomain({}{})'.format(self.domain, left + right)


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

    def __repr__(self):
        return f'CircleDomain({self.domain})'
