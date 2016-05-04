# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import ImmutableInterface
from pymor.domaindescriptions.boundarytypes import BoundaryType


class DomainDescriptionInterface(ImmutableInterface):
    """Describes a geometric domain along with its boundary.

    Attributes
    ----------
    dim
        The dimension of the domain
    boundary_types
        Set of |BoundaryTypes| the domain has.
    """

    dim = None
    boundary_types = frozenset()

    @property
    def has_dirichlet(self):
        return BoundaryType('dirichlet') in self.boundary_types

    @property
    def has_neumann(self):
        return BoundaryType('neumann') in self.boundary_types

    @property
    def has_robin(self):
        return BoundaryType('robin') in self.boundary_types
