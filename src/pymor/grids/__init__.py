from __future__ import absolute_import, division, print_function
from pymor.grids.interfaces import BoundaryInfoInterface
from pymor.grids.boundaryinfos import EmptyBoundaryInfo, BoundaryInfoFromIndicators, AllDirichletBoundaryInfo
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid
from pymor.grids.oned import OnedGrid

#import * from grid should not pull old reference implementations
__all__ = ['rect', 'tria', 'oned', 'interfaces', 'prescribed', 'referenceelements', 'defaultimpl']
