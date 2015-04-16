# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
import weakref


class WorkerPoolDefaultImplementations(object):

    @contextmanager
    def distribute_array(self, U, copy=True):

        def impl(U, copy):
            U = U()
            UR = U.empty()
            slice_len = len(U) // len(self) + (1 if len(U) % len(self) else 0)
            if copy:
                slices = []
                for i in range(len(self)):
                    slices.append(U.copy(ind=range(i*slice_len, min((i+1)*slice_len, len(U)))))
            else:
                slices = [U.empty() for _ in range(len(self))]
                for s in slices:
                    s.append(U, o_ind=range(0, min(slice_len, len(U))), remove_from_other=True)
            del U

            with self.distribute(UR) as remote_U:
                self.map(_append_array_slice, slices, U=remote_U)
                del slices
                yield remote_U

        # pass weakref to impl to make sure that no reference to U is kept while inside the context
        return impl(weakref.ref(U), copy)

    @contextmanager
    def distribute_list(self, l):

        def impl(l):
            slice_len = len(l) // len(self) + (1 if len(l) % len(self) else 0)
            slices = []
            for i in range(len(self)):
                slices.append(l[i*slice_len:(i+1)*slice_len])
            del l[:]

            with self.distribute([]) as remote_l:
                self.map(_append_list_slice, slices, l=remote_l)
                del slices
                yield remote_l

        # always pass a copy of l which we can empty inside impl so that we do not hold refs to the data
        # note that the weakref trick from 'distribute_array' does not work here since lists cannot be
        # weakrefed
        return impl(list(l))


def _append_array_slice(s, U=None):
    U.append(s, remove_from_other=True)


def _append_list_slice(s, l=None):
    l.extend(s)
