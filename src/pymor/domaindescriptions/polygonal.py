# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

import collections
from pymor.domaindescriptions.interfaces import DomainDescriptionInterface, KNOWN_BOUNDARY_TYPES


class PolygonalDomain(DomainDescriptionInterface):
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
        List of lists of points that describe the polygonal chains that bound the holes inside the domain.

    Attributes
    ----------
    points
    boundary_types
    holes
    """

    dim = 2

    def __init__(self, points, boundary_types, holes=None):
        holes = holes or []
        self.points = points
        self.holes = holes

        if isinstance(boundary_types, dict):
            self.boundary_types = boundary_types
        # if the boundary types are not given as a dict, try to evaluate at the edge centers to get a dict.
        else:
            points = [points]
            points.extend(holes)
            # shift points 1 entry to the left.
            points_deque = [collections.deque(ps) for ps in points]
            for ps_d in points_deque:
                ps_d.rotate(-1)
            # compute edge centers.
            centers = [[(p0[0]+p1[0])/2, (p0[1]+p1[1])/2] for ps, ps_d in zip(points, points_deque)
                       for p0, p1 in zip(ps, ps_d)]
            # evaluate the boundary at the edge centers and save the boundary types together with the
            # corresponding edge id.
            self.boundary_types = dict(zip([boundary_types(centers)], [list(range(1, len(centers)+1))]))

        for bt in self.boundary_types.keys():
            if bt is not None and bt not in KNOWN_BOUNDARY_TYPES:
                self.logger.warning(f'Unknown boundary type: {bt}')

    def __repr__(self):
        return f'PolygonalDomain({repr(self.points)}, {repr(self.boundary_types)}, {repr(self.holes)})'


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
        self.angle = angle
        self.radius = radius
        self.arc = arc
        self.radii = radii
        self.num_points = num_points
        assert (0 < self.angle) and (self.angle < 2*np.pi)
        assert self.radius > 0
        assert self.num_points > 0

        points = [[0., 0.]]
        points.extend([[self.radius*np.cos(t), self.radius*np.sin(t)] for t in
                       np.linspace(start=0, stop=angle, num=self.num_points, endpoint=True)])

        if self.arc == self.radii:
            boundary_types = {self.arc: list(range(1, len(points)+1))}
        else:
            boundary_types = {self.arc: list(range(2, len(points)))}
            boundary_types.update({self.radii: [1, len(points)]})

        if None in boundary_types:
            del boundary_types[None]

        super().__init__(points, boundary_types)

    def __repr__(self):
        return f'PieDomain({repr(self.angle)}, {repr(self.radius)}, {repr(self.arc)}, {repr(self.radii)}, ' \
               f'{repr(self.num_points)})'


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
        self.radius = radius
        self.boundary = boundary
        self.num_points = num_points
        assert self.radius > 0
        assert self.num_points > 0

        points = [[self.radius*np.cos(t), self.radius*np.sin(t)] for t in
                  np.linspace(start=0, stop=2*np.pi, num=num_points, endpoint=False)]
        boundary_types = {} if self.boundary is None else {boundary: list(range(1, len(points)+1))}

        super().__init__(points, boundary_types)

    def __repr__(self):
        return f'DiscDomain({repr(self.radius)}, {repr(self.boundary)}, {repr(self.num_points)})'
