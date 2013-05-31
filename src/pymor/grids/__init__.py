# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
from pymor.grids.interfaces import BoundaryInfoInterface
from pymor.grids.boundaryinfos import EmptyBoundaryInfo, BoundaryInfoFromIndicators, AllDirichletBoundaryInfo
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid
from pymor.grids.oned import OnedGrid
from pymor.grids.subgrid import SubGrid

#import * from grid should not pull old reference implementations
__all__ = ['rect', 'tria', 'oned', 'subgrid', 'interfaces', 'referenceelements', 'defaultimpl']
