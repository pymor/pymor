# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.tools import mpi
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.mpi import MPIVectorArray


class MPIDiscretization(DiscretizationInterface):

    def __init__(self, obj_id):
        impl = mpi.get_object(obj_id)
        self.name = impl.name
        self.obj_id = obj_id
        self.solution_space = VectorSpace(MPIVectorArray, (impl.solution_space.type, impl.solution_space.subtype))
        self.cache_region = None

    def _solve(self, mu=None):
        impl_solution_space = self.solution_space.subtype
        return MPIVectorArray(impl_solution_space.type, impl_solution_space.subtype,
                              mpi.call(mpi.method_call_manage, self.obj_id, 'solve', mu=mu))

    def visualize(self, U):
        assert U in self.solution_space
        mpi.call(mpi.method_call1, self.obj_id, 'visualize', U.obj_id)
