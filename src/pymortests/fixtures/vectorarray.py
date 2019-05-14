# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from itertools import product
import numpy as np
import pytest

from pymor.core.config import config
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.list import NumpyListVectorSpace


import os

def random_integers(count, seed):
    np.random.seed(seed)
    return list(np.random.randint(0, 3200, count))


def numpy_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    if np.random.randint(2):
        return NumpyVectorSpace.from_numpy(np.random.random((length, dim)))
    else:
        return NumpyVectorSpace.from_numpy(np.random.random((length, dim)) + np.random.random((length, dim)) * 1j)


def numpy_list_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    if np.random.randint(2):
        return NumpyListVectorSpace.from_numpy(np.random.random((length, dim)))
    else:
        return NumpyListVectorSpace.from_numpy(np.random.random((length, dim)) + np.random.random((length, dim)) * 1j)


def block_vector_array_factory(length, dims, seed):
    return BlockVectorSpace([NumpyVectorSpace(dim) for dim in dims]).from_numpy(
        numpy_vector_array_factory(length, sum(dims), seed).to_numpy()
    )

if config.HAVE_FENICS:
    import dolfin as df
    from pymor.bindings.fenics import FenicsVectorSpace

    fenics_spaces = [df.FunctionSpace(df.UnitSquareMesh(ni, ni), 'Lagrange', 1)
                     for ni in [1, 10, 32, 100]]

    def fenics_vector_array_factory(length, space, seed):
        V = FenicsVectorSpace(fenics_spaces[space])
        U = V.zeros(length)
        dim = V.dim
        np.random.seed(seed)
        for v, a in zip(U._list, np.random.random((length, dim))):
            v.real_part.impl[:] = a
        if np.random.randint(2):
            UU = V.zeros(length)
            for v, a in zip(UU._list, np.random.random((length, dim))):
                v.real_part.impl[:] = a
            for u, uu in zip(U._list, UU._list):
                u.imag_part = uu.real_part
        return U

    fenics_vector_array_factory_arguments = \
        list(zip([0,  0,  1, 43, 102],      # len
            [0,  1,  3,  2,  2],      # ni
            random_integers(5, 123)))   # seed

    fenics_vector_array_factory_arguments_pairs_with_same_dim = \
        list(zip([0,  0,   1, 43, 102,  2],         # len1
            [0,  1,  37,  9, 104,  2],         # len2
            [0,  1,   3,  2,   2,  2],         # dim
            random_integers(5, 1234) + [42],  # seed1
            random_integers(5, 1235) + [42]))  # seed2

    fenics_vector_array_factory_arguments_pairs_with_different_dim = \
        list(zip([0,  0,  1, 43, 102],      # len1
            [0,  1,  1,  9,  10],      # len2
            [0,  1,  2,  3,   1],      # dim1
            [1,  2,  1,  2,   3],      # dim2
            random_integers(5, 1234),  # seed1
            random_integers(5, 1235)))  # seed2


if config.HAVE_NGSOLVE:
    import ngsolve as ngs
    import netgen.meshing as ngmsh
    from netgen.geom2d import unit_square
    from pymor.bindings.ngsolve import NGSolveVectorSpace

    NGSOLVE_spaces = {}

    def ngsolve_vector_array_factory(length, dim, seed):

        if dim not in NGSOLVE_spaces:
            mesh = ngmsh.Mesh(dim=1)
            if dim > 0:
                pids = []
                for i in range(dim+1):
                    pids.append(mesh.Add(ngmsh.MeshPoint(ngmsh.Pnt(i/dim, 0, 0))))
                for i in range(dim):
                    mesh.Add(ngmsh.Element1D([pids[i], pids[i+1]], index=1))

            NGSOLVE_spaces[dim] = NGSolveVectorSpace(ngs.L2(ngs.Mesh(mesh), order=0))

        U = NGSOLVE_spaces[dim].zeros(length)
        np.random.seed(seed)
        for v, a in zip(U._list, np.random.random((length, dim))):
            v.to_numpy()[:] = a
        if np.random.randint(2):
            UU = NGSOLVE_spaces[dim].zeros(length)
            for v, a in zip(UU._list, np.random.random((length, dim))):
                v.real_part.to_numpy()[:] = a
            for u, uu in zip(U._list, UU._list):
                u.imag_part = uu.real_part
        return U


if config.HAVE_DEALII:
    from pydealii.pymor.vectorarray import DealIIVectorSpace

    def dealii_vector_array_factory(length, dim, seed):
        U = DealIIVectorSpace(dim).zeros(length)
        np.random.seed(seed)
        for v, a in zip(U._list, np.random.random((length, dim))):
            v.impl[:] = a
        return U


def vector_array_from_empty_reserve(v, reserve):
    if reserve == 0:
        return v
    if reserve == 1:
        r = 0
    elif reserve == 2:
        r = len(v) + 10
    elif reserve == 3:
        r = int(len(v) / 2)
    c = v.empty(reserve=r)
    c.append(v)
    return c


