# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('FENICS')

from dolfin import RectangleMesh, Point, SubDomain, MeshFunction, DOLFIN_EPS
import numpy as np

from pymor.analyticalproblems.domaindescriptions import (RectDomain, CylindricalDomain, TorusDomain, LineDomain,
                                                         CircleDomain, PolygonalDomain)


def discretize_domain(domain_description, diameter=1 / 100):

    def discretize_RectDomain():
        x0i = int(np.ceil(domain_description.width * np.sqrt(2) / diameter))
        x1i = int(np.ceil(domain_description.height * np.sqrt(2) / diameter))
        mesh = RectangleMesh(
            Point(domain_description.domain[0]),
            Point(domain_description.domain[1]),
            x0i,
            x1i
        )

        class RectBoundaries(SubDomain):

            def __init__(self, boundary_type):
                self.boundary_type = boundary_type
                super().__init__()

            def inside(self, x, on_boundary):
                bt = self.boundary_type
                if not on_boundary:
                    return False
                if (x[0] < domain_description.domain[0, 0] + DOLFIN_EPS) and domain_description.left == bt:
                    return True
                if (x[0] > domain_description.domain[1, 0] - DOLFIN_EPS) and domain_description.right == bt:
                    return True
                if (x[1] < domain_description.domain[0, 1] + DOLFIN_EPS) and domain_description.bottom == bt:
                    return True
                if (x[1] > domain_description.domain[1, 1] - DOLFIN_EPS) and domain_description.top == bt:
                    return True
                return False

        boundary_mask = MeshFunction('size_t', mesh, 1)
        boundary_mask.set_all(0)
        boundary_ids = {bt: i for i, bt in enumerate(domain_description.boundary_types, start=1)}
        for bt, i in boundary_ids.items():
            rb = RectBoundaries(bt)
            rb.mark(boundary_mask, i)

        return mesh, (boundary_mask, boundary_ids)

    if isinstance(domain_description, RectDomain):
        return discretize_RectDomain()
    elif isinstance(domain_description, CylindricalDomain):
        raise NotImplementedError
    elif isinstance(domain_description, TorusDomain):
        raise NotImplementedError
    elif isinstance(domain_description, PolygonalDomain):
        raise NotImplementedError
    elif isinstance(domain_description, LineDomain):
        return NotImplementedError
    elif isinstance(domain_description, CircleDomain):
        raise NotImplementedError
    else:
        raise NotImplementedError
