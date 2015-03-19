# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Schaefer <michael.schaefer@uni-muenster.de>

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import ImmutableInterface
from pymor.domaindescriptions.boundarytypes import BoundaryType


class DomainDescriptionInterface(ImmutableInterface):
    """Describes a geometric domain along with its boundary.

    Attributes
    ----------
    boundary_types
        Set of |BoundaryTypes| the domain has.
    """

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
