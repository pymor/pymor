from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import numpy as np

# For python3.2 and greater, we user functools.lru_cache for caching. If our python
# version is to old, we import the same decorator from the third-party functools32
# package
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

from .interfaces import IGrid
from .exceptions import CodimError


class Base(IGrid):
    '''An abstract base class for grids implementing superentities() and
    neighbours() in terms of subentities(). Also, subentities(c, sc) is
    computed in terms of subentities(c, None).

    All results are cached for the next evaluation using functools.lru_cache.
    Note that the current implementation is very slow and should be moved to
    C code. For g = pymor.grid.rect.Rect((1000, 1000)) it takes around 4.5
    minutes on modern hardware to calculate g.neighbours(0, 1, 2).
    (The call involves calculating g.subentites(0, 2) and g.superentities(2, 1)
    at the same time.)
    '''

    @abc.abstractmethod
    @lru_cache(maxsize=None)
    def subentities(self, codim=0, subentity_codim=None):
        assert 0 <= codim < self.dim, CodimError('Invalid codimension')
        if subentity_codim > codim + 1:
            SE = self.subentities(codim, subentity_codim - 1)
            SESE = self.subentities(subentity_codim - 1, subentity_codim)

            # we assume that there is only one geometry type ...
            num_subsubentities = np.unique(SESE[:, SE[:, 0]]).size

            SSE = np.empty((num_subsubentities, SE.shape[1]), dtype=np.int32)
            SSE.fill(-1)

            for ei in xrange(SE.shape[1]):
                X = SESE[:, SE[:, ei]].T.ravel()
                SSE[:, ei] = X[np.sort(np.unique(X, return_index=True)[1])]

            return SSE
        else:
            raise NotImplementedError

    @lru_cache(maxsize=None)
    def superentities(self, codim, superentity_codim=None):
        assert 0 < codim <= self.dim, CodimError('Invalid codimension')
        if superentity_codim is None:
            superentity_codim = codim - 1

        SE = self.subentities(superentity_codim, codim)
        num_superentities = np.bincount(SE.ravel()).max()
        SPE = np.empty((num_superentities, self.size(codim)), dtype=np.int32)
        SPE.fill(-1)

        SPE_COUNTS = np.zeros(SPE.shape[1], dtype=np.int32)

        for index, se in np.ndenumerate(SE):
            if se >= 0:
                SPE[SPE_COUNTS[se], se] = index[1]
                SPE_COUNTS[se] += 1

        return SPE

    @lru_cache(maxsize=None)
    def neighbours(self, codim=0, neighbour_codim=0, intersection_codim=None):
        if intersection_codim is None:
            if codim == neighbour_codim:
                intersection_codim = codim + 1
            else:
                intersection_codim = min(codim, neighbour_codim)

        if intersection_codim == min(codim, neighbour_codim):
            if codim <= neighbour_codim:
                return self.subentities(codim, neighbour_codim)
            else:
                return self.superentities(codim, neighbour_codim)
        else:
            EI = self.subentities(codim, intersection_codim)
            ISE = self.superentities(intersection_codim, neighbour_codim)

            NB = np.empty((EI.shape[0] * ISE.shape[0], EI.shape[1]), dtype=np.int32)
            NB.fill(-1)
            NB_COUNTS = np.zeros(EI.shape[1], dtype=np.int32)

            if codim == neighbour_codim:
                for ii, i in np.ndenumerate(EI):
                    if i >= 0:
                        for ni, n in np.ndenumerate(ISE[:, i]):
                            if n != ii[1] and n not in NB[:, ii[1]]:
                                NB[NB_COUNTS[ii[1]], ii[1]] = n
                                NB_COUNTS[ii[1]] += 1
            else:
                for ii, i in np.ndenumerate(EI):
                    if i >= 0:
                        for ni, n in np.ndenumerate(ISE[:, i]):
                            if n not in NB[:, ii[1]]:
                                NB[NB_COUNTS[ii[1]], ii[1]] = n
                                NB_COUNTS[ii[1]] += 1

            NB.resize((NB_COUNTS.max(), NB.shape[1]))
            return NB
