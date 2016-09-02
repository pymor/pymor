# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
import atexit
from collections import OrderedDict
import tempfile
import getpass
import os
import shutil
import numpy as np

from pymor.core.defaults import defaults
from pymor.core.pickle import dump, load
from pymor.vectorarrays.interfaces import VectorArrayInterface, _INDEXTYPES
from pymor.vectorarrays.list import ListVectorArray


_registered_paths = set()
def cleanup():
    for path in _registered_paths:
        shutil.rmtree(path)
atexit.register(cleanup)


@defaults('path', sid_ignore=('path',))
def basedir(path=os.path.join(tempfile.gettempdir(), 'pymor.diskarray.' + getpass.getuser())):
    if not os.path.exists(path):
        os.mkdir(path)
    _registered_paths.add(path)
    return path


class DiskVectorArray(VectorArrayInterface):
    """|VectorArray| implementation via a list of vectors stored in temporary files."""

    _NONE = ()

    def __init__(self, vectors, subtype=_NONE):
        if isinstance(vectors, ListVectorArray):
            assert subtype is self._NONE \
                or (type(subtype) is tuple and len(subtype) == 2
                    and subtype[0] == vectors.vector_type and subtype[1] == vectors.subtype)
            subtype = (vectors.vector_type and vectors.subtype)
            vectors = vectors._list
        else:
            vectors = list(vectors)
        if subtype is self._NONE:
            assert len(vectors) > 0
            subtype = (type(vectors[0]), vectors[0].subtype)
        assert all(type(v) == subtype[0] and v.subtype == subtype[1] for v in vectors)
        self.vector_type, self.vector_subtype = subtype
        self.dir = tempfile.mkdtemp(dir=basedir())
        self._len = len(vectors)
        self.cache_size = 1
        self._cache = OrderedDict()
        for i, v in enumerate(vectors):
            self._store(i, v)

    def __del__(self):
        self.destroy()

    def destroy(self):
        try:
            shutil.rmtree(self.dir)
        except OSError:
            pass

    def _store(self, i, v):
        with open(os.path.join(self.dir, str(i)), 'wb') as f:
            dump(v, f)
        self._cache[i] = v
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _load(self, i):
        if i in self._cache:
            return self._cache[i]
        else:
            with open(os.path.join(self.dir, str(i)), 'rb') as f:
                v = load(f)
            self._cache[i] = v
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return v

    def get(self, ind=None):
        assert self.check_ind(ind)
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        return ListVectorArray([self._load(i) for i in ind], subtype=self.subtype, copy=False)

    @classmethod
    def make_array(cls, subtype=None, count=0, reserve=0):
        assert count >= 0
        assert reserve >= 0
        va = cls([], subtype)
        v = va.vector_type.make_zeros(va.vector_subtype)
        for i in range(count):
            va._store(i, v)
        va._len = count
        return va

    @classmethod
    def from_data(cls, data, subtype):
        va = cls([], subtype)
        for i in range(len(data)):
            v = va.vector_type.from_data(data[i], va.vector_subtype)
            va._store(i, v)
        return va

    def __len__(self):
        return self._len

    @property
    def dim(self):
        return self.vector_type.make_zeros(self.vector_subtype).dim

    @property
    def subtype(self):
        return (self.vector_type, self.vector_subtype)

    def copy(self, ind=None):
        assert self.check_ind(ind)
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        c = type(self)([], subtype=self.subtype)
        for d, s in enumerate(ind):
            shutil.copy(os.path.join(self.dir, str(s)),
                        os.path.join(c.dir, str(d)))
        c._len = len(ind)
        return c

    def append(self, other, o_ind=None, remove_from_other=False):
        assert other.check_ind(o_ind)
        assert other.space == self.space
        assert other is not self or not remove_from_other
        o_ind = list(range(other._len)) if o_ind is None else [o_ind] if isinstance(o_ind, Number) else o_ind

        self._cache.clear()
        if not remove_from_other:
            for d, s in enumerate(o_ind):
                shutil.copy(os.path.join(other.dir, str(s)),
                            os.path.join(self.dir, str(d + self._len)))
            self._len += len(o_ind)
        else:
            other._cache.clear()
            if len(set(o_ind)) < len(o_ind):
                self.append(other, o_ind=o_ind, remove_from_other=False)
                other.remove(o_ind)
            else:
                for d, s in enumerate(o_ind):
                    shutil.move(os.path.join(other.dir, str(s)),
                                os.path.join(self.dir, str(d + self._len)))
                self._len += len(o_ind)

                remaining = sorted(set(range(other._len)) - set(o_ind))
                for d, s in enumerate(remaining):
                    shutil.move(os.path.join(other.dir, str(s)),
                                os.path.join(other.dir, str(d)))
                other._len = len(remaining)

    def remove(self, ind=None):
        assert self.check_ind(ind)
        self._cache.clear()
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else list(set(ind))
        for i in ind:
            os.remove(os.path.join(self.dir, str(i)))
        remaining = sorted(set(range(len(self))) - set(ind))
        for d, s in enumerate(remaining):
            shutil.move(os.path.join(self.dir, str(s)),
                        os.path.join(self.dir, str(d)))
        self._len = len(remaining)

    def scal(self, alpha, ind=None):
        assert self.check_ind_unique(ind)
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(ind),)
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        if isinstance(alpha, np.ndarray):
            for a, i in zip(alpha, ind):
                new_vec = self._load(i)
                new_vec.scal(a)
                self._store(i, new_vec)
        else:
            for i in ind:
                new_vec = self._load(i)
                new_vec.scal(alpha)
                self._store(i, new_vec)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        assert self.check_ind_unique(ind)
        assert x.check_ind(x_ind)
        assert self.space == x.space
        assert self.len_ind(ind) == x.len_ind(x_ind) or x.len_ind(x_ind) == 1
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self.len_ind(ind),)
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        x_ind = list(range(x._len)) if x_ind is None else [x_ind] if isinstance(x_ind, _INDEXTYPES) else x_ind

        if self is x:
            if set(ind).intersection(set(x_ind)):
                self.axpy(alpha, x.copy(x_ind), ind)
                return

        if np.all(alpha == 0):
            return

        if len(x_ind) == 1:
            x = x._load(x_ind[0])
            if isinstance(alpha, np.ndarray):
                for a, y in zip(alpha, ind):
                    new_vec = self._load(y)
                    new_vec.axpy(a, x)
                    self._store(y, new_vec)
            else:
                for y in ind:
                    new_vec = self._load(y)
                    new_vec.axpy(alpha, x)
                    self._store(y, new_vec)
        else:
            if isinstance(alpha, np.ndarray):
                for a, xx, y in zip(alpha, x_ind, ind):
                    new_vec = self._load(y)
                    new_vec.axpy(a, x._load(xx))
                    self._store(y, new_vec)
            else:
                for xx, y in zip(x_ind, ind):
                    new_vec = self._load(y)
                    new_vec.axpy(alpha, x._load(xx))
                    self._store(y, new_vec)

    def dot(self, other, ind=None, o_ind=None):
        assert self.check_ind(ind)
        assert other.check_ind(o_ind)
        assert self.space == other.space
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        o_ind = list(range(other._len)) if o_ind is None else [o_ind] if isinstance(o_ind, Number) else o_ind

        R = np.empty((len(ind), len(o_ind)))
        for i, a in enumerate(ind):
            for j, b in enumerate(o_ind):
                R[i, j] = self._load(a).dot(other._load(b))
        return R

    def pairwise_dot(self, other, ind=None, o_ind=None):
        assert self.check_ind(ind)
        assert other.check_ind(o_ind)
        assert self.space == other.space
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        o_ind = list(range(other._len)) if o_ind is None else [o_ind] if isinstance(o_ind, Number) else o_ind
        assert len(ind) == len(o_ind)
        return np.array([self._load(i).dot(other._load(oi)) for i, oi in zip(ind, o_ind)])

    def gramian(self, ind=None):
        assert self.check_ind(ind)
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind

        R = np.empty((len(ind), len(ind)))
        for i, ii in enumerate(ind):
            for j, jj in enumerate(ind[i:], i):
                R[i, j] = self._load(ii).dot(self._load(jj))
                R[j, i] = R[i, j]
        return R

    def lincomb(self, coefficients, ind=None):
        assert self.check_ind(ind)
        assert 1 <= coefficients.ndim <= 2
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind

        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, :]
        assert coefficients.shape[1] == self.len_ind(ind)

        RL = type(self)([], subtype=self.subtype)
        for coeffs in coefficients:
            R = self.vector_type.make_zeros(self.vector_subtype)
            for i, c in zip(ind, coeffs):
                R.axpy(c, self._load(i))
            RL._store(RL._len, R)
            RL._len += 1

        return RL

    def l1_norm(self, ind=None):
        assert self.check_ind(ind)
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        return np.array([self._load(i).l1_norm() for i in ind])

    def l2_norm(self, ind=None):
        assert self.check_ind(ind)
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        return np.array([self._load(i).l2_norm() for i in ind])

    def l2_norm2(self, ind=None):
        assert self.check_ind(ind)
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind
        return np.array([self._load(i).l2_norm2() for i in ind])

    def components(self, component_indices, ind=None):
        assert self.check_ind(ind)
        assert isinstance(component_indices, list) and (len(component_indices) == 0 or min(component_indices) >= 0) \
            or (isinstance(component_indices, np.ndarray) and component_indices.ndim == 1
                and (len(component_indices) == 0 or np.min(component_indices) >= 0))
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind

        if len(ind) == 0:
            assert len(component_indices) == 0 \
                or isinstance(component_indices, list) and max(component_indices) < self.dim \
                or isinstance(component_indices, np.ndarray) and np.max(component_indices) < self.dim
            return np.empty((0, len(component_indices)))

        R = np.empty((len(ind), len(component_indices)))
        for k, i in enumerate(ind):
            R[k] = self._load(i).components(component_indices)

        return R

    def amax(self, ind=None):
        assert self.check_ind(ind)
        assert self.dim > 0
        ind = list(range(self._len)) if ind is None else [ind] if isinstance(ind, Number) else ind

        MI = np.empty(len(ind), dtype=np.int)
        MV = np.empty(len(ind))

        for k, i in enumerate(ind):
            MI[k], MV[k] = self._load(i).amax()

        return MI, MV

    def __str__(self):
        return 'DiskVectorArray of {} {}s of dimension {}'.format(len(self._len), str(self.vector_type), self.dim)
