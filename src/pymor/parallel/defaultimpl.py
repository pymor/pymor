# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function


class WorkerPoolDefaultImplementations(object):

    def scatter_array(self, U, copy=True):
        slice_len = len(U) // len(self) + (1 if len(U) % len(self) else 0)
        if copy:
            slices = []
            for i in range(len(self)):
                slices.append(U.copy(ind=range(i*slice_len, min((i+1)*slice_len, len(U)))))
        else:
            slices = [U.empty() for _ in range(len(self))]
            for s in slices:
                s.append(U, o_ind=range(0, min(slice_len, len(U))), remove_from_other=True)
        remote_U = self.push(U.empty())
        del U
        self.map(_append_array_slice, slices, U=remote_U)
        return remote_U

    def scatter_list(self, l):
        slice_len = len(l) // len(self) + (1 if len(l) % len(self) else 0)
        slices = []
        for i in range(len(self)):
            slices.append(l[i*slice_len:(i+1)*slice_len])
        del l
        remote_l = self.push([])
        self.map(_append_list_slice, slices, l=remote_l)
        return remote_l


def _append_array_slice(s, U=None):
    U.append(s, remove_from_other=True)


def _append_list_slice(s, l=None):
    l.extend(s)
