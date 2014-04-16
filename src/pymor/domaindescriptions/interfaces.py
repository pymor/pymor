# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core import ImmutableInterface
from pymor.domaindescriptions.boundarytypes import BoundaryType


class DomainDescriptionInterface(ImmutableInterface):
    '''Analytically describes a domain and its boundary (types).

    Attributes
    ----------
    boundary_types
        Set of |BoundaryTypes| the domain has.
    '''

    boundary_types = frozenset()

    @property
    def has_dirichlet(self):
        return BoundaryType('dirichlet') in self.boundary_types

    @property
    def has_neumann(self):
        return BoundaryType('neumann') in self.boundary_types

    @property
    def has_only_dirichlet(self):
        return self.boundary_types == {BoundaryType('dirichlet')}

    @property
    def has_only_neumann(self):
        return self.boundary_types == {BoundaryType('neumann')}

    @property
    def has_only_dirichletneumann(self):
        return self.boundary_types <= {BoundaryType('dirichlet'), BoundaryType('neumann')}
