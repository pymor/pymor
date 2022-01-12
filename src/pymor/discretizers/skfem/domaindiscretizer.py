# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('SCIKIT_FEM')


import numpy as np
import skfem

from pymor.analyticalproblems.domaindescriptions import (RectDomain, CylindricalDomain, TorusDomain, LineDomain,
                                                         CircleDomain, PolygonalDomain)


def discretize_domain(domain_description, diameter=1 / 100, mesh_type=None):

    def discretize_RectDomain():
        mt = mesh_type or skfem.MeshQuad
        x0i = int(np.ceil(domain_description.width * np.sqrt(2) / diameter))
        x1i = int(np.ceil(domain_description.height * np.sqrt(2) / diameter))
        mesh = mt.init_tensor(
            np.linspace(domain_description.domain[0, 0], domain_description.domain[1, 0], x0i + 1),
            np.linspace(domain_description.domain[0, 1], domain_description.domain[1, 1], x1i + 1)
        ).with_boundaries({
            'left': lambda x: x[0] == domain_description.domain[0, 0],
            'right': lambda x: x[0] == domain_description.domain[1, 0],
            'top': lambda x: x[1] == domain_description.domain[1, 1],
            'bottom': lambda x: x[1] == domain_description.domain[0, 1],
        })

        boundary_facets = {
            bt: np.hstack([mesh.boundaries[edge]
                          for edge in ['left', 'right', 'top', 'bottom']
                          if getattr(domain_description, edge) == bt])
            for bt in domain_description.boundary_types
        }

        return mesh, boundary_facets

    def discretize_LineDomain():
        mt = mesh_type or skfem.MeshLine
        mesh = mt(
            np.linspace(domain_description.domain[0], domain_description.domain[1],
                        int(np.ceil((domain_description.domain[1] - domain_description.domain[0]) / diameter)))
        ).with_boundaries({
            'left': lambda x: x[0] == domain_description.domain[0],
            'right': lambda x: x[0] == domain_description.domain[1]
        })

        boundary_facets = {
            bt: np.hstack([mesh.boundaries[edge]
                          for edge in ['left', 'right']
                          if getattr(domain_description, edge) == bt])
            for bt in domain_description.boundary_types
        }

        return mesh, boundary_facets

    if isinstance(domain_description, RectDomain):
        return discretize_RectDomain()
    elif isinstance(domain_description, CylindricalDomain):
        raise NotImplementedError
    elif isinstance(domain_description, TorusDomain):
        raise NotImplementedError
    elif isinstance(domain_description, PolygonalDomain):
        # from pymor.discretizers.builtin.domaindiscretizers.gmsh import discretize_gmsh
        # return discretize_gmsh(domain_description, clscale=diameter)
        raise NotImplementedError
    elif isinstance(domain_description, LineDomain):
        return discretize_LineDomain()
    elif isinstance(domain_description, CircleDomain):
        raise NotImplementedError
    else:
        raise NotImplementedError
