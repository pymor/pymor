# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import division

import numpy as np
import cython
cimport numpy as np

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t


@cython.boundscheck(False)
def compute_edges(np.ndarray[DTYPE_t, ndim=2] faces, int num_vertices):
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int m
    cdef unsigned int n
    cdef int x
    cdef unsigned int index
    cdef unsigned int max_edges

    cdef np.ndarray[DTYPE_t, ndim=1] edge_counts
    edge_counts = np.zeros(num_vertices, dtype=DTYPE)

    for i in xrange(faces.shape[0]):
        for j in xrange(faces.shape[1]):
            edge_counts[<unsigned int>faces[i, j]] += 2

    max_edges = np.max(edge_counts)
    del edge_counts

    cdef unsigned int edge_counter
    cdef np.ndarray[DTYPE_t, ndim=2] edge_memo_vertices
    cdef np.ndarray[DTYPE_t, ndim=2] edge_memo_ids
    cdef np.ndarray[DTYPE_t, ndim=2] edges
    edge_memo_vertices = np.empty((num_vertices, max_edges), dtype=DTYPE)
    edge_memo_vertices[:] = -1
    edge_memo_ids= np.empty((num_vertices, max_edges), dtype=DTYPE)
    edges = np.empty_like(faces)

    edge_counter = 0
    for i in xrange(faces.shape[0]):
        for j in xrange(3):
            if j == 0:
                m = faces[i, <unsigned int>1]
                n = faces[i, <unsigned int>2]
            elif j == 1:
                m = faces[i, <unsigned int>2]
                n = faces[i, <unsigned int>0]
            else:
                m = faces[i, <unsigned int>0]
                n = faces[i, <unsigned int>1]

            if m > n:
                m, n = n, m

            # try to find edge in memo
            for index in xrange(max_edges):
                x = edge_memo_vertices[m, index]
                if x == n or x == -1:
                    break

            if x == n:
                edges[i, j] = edge_memo_ids[m, index]
            else:
                assert x == -1
                edge_memo_ids[m, index] = edges[i, j] = edge_counter
                edge_memo_vertices[m, index] = n
                edge_counter += 1

    return edges, edge_counter
