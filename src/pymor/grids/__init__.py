# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.grids.boundaryinfos import EmptyBoundaryInfo, BoundaryInfoFromIndicators, AllDirichletBoundaryInfo
from pymor.grids.interfaces import BoundaryInfoInterface
from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.grids.subgrid import SubGrid
from pymor.grids.tria import TriaGrid

#import * from grid should not pull old reference implementations
__all__ = ['rect', 'tria', 'oned', 'subgrid', 'interfaces', 'referenceelements', 'defaultimpl']
