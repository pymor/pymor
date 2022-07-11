# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from collections import defaultdict
from itertools import chain

import numpy as np

from pymor.core.base import ImmutableObject


KNOWN_BOUNDARY_TYPES = {'dirichlet', 'neumann', 'robin'}


class DomainDescription(ImmutableObject):
    """Describes a geometric domain along with its boundary.

    Attributes
    ----------
    dim
        The dimension of the domain
    boundary_types
        Set of boundary types the domain has.
    """

    dim = None
    boundary_types = frozenset()

    @property
    def has_dirichlet(self):
        return 'dirichlet' in self.boundary_types

    @property
    def has_neumann(self):
        return 'neumann' in self.boundary_types

    @property
    def has_robin(self):
        return 'robin' in self.boundary_types


class RectDomain(DomainDescription):
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


class CylindricalDomain(DomainDescription):
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


class TorusDomain(DomainDescription):
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


class LineDomain(DomainDescription):
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


class CircleDomain(DomainDescription):
    """Domain with the topology of a circle, i.e. a line with end points identified.

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


class PolygonalDomain(DomainDescription):
    """Describes a domain with a polygonal boundary and polygonal holes inside the domain.

    Parameters
    ----------
    points
        List of points [x_0, x_1] that describe the polygonal chain that bounds the domain.
    boundary_types
        Either a dictionary `{boundary_type: [i_0, ...], boundary_type: [j_0, ...], ...}`
        with `i_0, ...` being the ids of boundary segments for a given boundary type
        (`0` is the line connecting point `0` to `1`, `1` is the line connecting point `1` to `2`
        etc.), or a function that returns the boundary type for a given coordinate.
    holes
        List of lists of points that describe the polygonal chains that bound the holes
        inside the domain.

    Attributes
    ----------
    points
    boundary_types
    holes
    """

    dim = 2

    def __init__(self, points, boundary_types, holes=None):
        holes = holes or []

        if isinstance(boundary_types, dict):
            pass
        # if the boundary types are not given as a dict, try to evaluate at
        # the edge centers to get a dict.
        else:
            segment_id = 0
            boundary_types_dict = defaultdict(list)
            for curve in chain([points], holes):
                for index in range(len(curve)):
                    p0, p1 = curve[index], curve[index % len(curve)]
                    center = [(p0[0]+p1[0])/2, (p0[1]+p1[1])/2]
                    boundary_types_dict[boundary_types(center)].append(segment_id)
                    segment_id += 1

            boundary_types = dict(boundary_types_dict)

        for bt in boundary_types.keys():
            if bt is not None and bt not in KNOWN_BOUNDARY_TYPES:
                self.logger.warning(f'Unknown boundary type: {bt}')

        self.__auto_init(locals())


class CircularSectorDomain(PolygonalDomain):
    """Describes a circular sector domain of variable radius.

    Parameters
    ----------
    angle
        The angle between 0 and 2*pi of the circular sector.
    radius
        The radius of the circular sector.
    arc
        The boundary type of the arc.
    radii
        The boundary type of the two radii.
    num_points
        The number of points of the polygonal chain approximating the circular
        boundary.

    Attributes
    ----------
    angle
    radius
    arc
    radii
    num_points
    """

    def __init__(self, angle, radius, arc='dirichlet', radii='dirichlet', num_points=100):
        assert (0 < angle) and (angle < 2*np.pi)
        assert radius > 0
        assert num_points > 0

        points = [[0., 0.]]
        points.extend([[radius*np.cos(t), radius*np.sin(t)] for t in
                       np.linspace(start=0, stop=angle, num=num_points, endpoint=True)])

        if arc == radii:
            boundary_types = {arc: list(range(1, len(points)+1))}
        else:
            boundary_types = {arc: list(range(2, len(points)))}
            boundary_types.update({radii: [1, len(points)]})

        if None in boundary_types:
            del boundary_types[None]

        super().__init__(points, boundary_types)
        self.__auto_init(locals())


class DiscDomain(PolygonalDomain):
    """Describes a disc domain of variable radius.

    Parameters
    ----------
    radius
        The radius of the disc.
    boundary
        The boundary type of the boundary.
    num_points
        The number of points of the polygonal chain approximating the boundary.

    Attributes
    ----------
    radius
    boundary
    num_points
    """

    def __init__(self, radius, boundary='dirichlet', num_points=100):
        assert radius > 0
        assert num_points > 0

        points = [[radius*np.cos(t), radius*np.sin(t)] for t in
                  np.linspace(start=0, stop=2*np.pi, num=num_points, endpoint=False)]
        boundary_types = {} if boundary is None else {boundary: list(range(1, len(points)+1))}

        super().__init__(points, boundary_types)
        self.__auto_init(locals())
