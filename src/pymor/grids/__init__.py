from .interfaces import BoundaryInfoInterface
from .boundaryinfos import BoundaryInfoFromIndicators, AllDirichletBoundaryInfo

#import * from grid should not pull old reference implementations
__all__ = ['rect', 'tria', 'interfaces', 'prescribed', 'referenceelements', 'defaultimpl']
