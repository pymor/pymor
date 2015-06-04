# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import ImmutableInterface
from pymor.discretizations.basic import DiscretizationBase
from pymor.operators.mpi import mpi_wrap_operator
from pymor.tools import mpi
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.mpi import MPIVectorArray


class MPIDiscretization(DiscretizationBase):

    def __init__(self, obj_id, operators, functionals, vector_operators, products=None, array_type=MPIVectorArray):
        d = mpi.get_object(obj_id)
        visualizer = MPIVisualizer(obj_id)
        super(MPIDiscretization, self).__init__(operators, functionals, vector_operators, products=products,
                                                visualizer=visualizer, cache_region=None, name=d.name)
        self.obj_id = obj_id
        subtypes = mpi.call(_MPIDiscretization_get_subtypes, obj_id)
        if all(subtype == subtypes[0] for subtype in subtypes):
            subtypes = (subtypes[0],)
        self.solution_space = VectorSpace(array_type, (d.solution_space.type, subtypes))
        self.build_parameter_type(inherits=(d,))

    def _solve(self, mu=None):
        space = self.solution_space
        return space.type(space.subtype[0], space.subtype[1],
                          mpi.call(mpi.method_call_manage, self.obj_id, 'solve', mu=mu))

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


def _MPIDiscretization_get_subtypes(self):
    self = mpi.get_object(self)
    subtypes = mpi.comm.gather(self.solution_space.subtype, root=0)
    if mpi.rank0:
        return tuple(subtypes)


class MPIVisualizer(ImmutableInterface):

    def __init__(self, d_obj_id):
        self.d_obj_id = d_obj_id

    def visualize(self, U, d, **kwargs):
        mpi.call(mpi.method_call, self.d_obj_id, 'visualize', U.obj_id, **kwargs)


def mpi_wrap_discretization(obj_id, use_with=False, with_apply2=False, array_type=MPIVectorArray):

    operators, functionals, vectors, products = \
        mpi.call(_mpi_wrap_discretization_manage_operators, obj_id)

    operators = {k: mpi_wrap_operator(v, with_apply2=with_apply2, array_type=array_type) if v else None
                 for k, v in operators.iteritems()}
    functionals = {k: mpi_wrap_operator(v, functional=True, with_apply2=with_apply2, array_type=array_type) if v else None
                   for k, v in functionals.iteritems()}
    vectors = {k: mpi_wrap_operator(v, vector=True, with_apply2=with_apply2, array_type=array_type) if v else None
               for k, v in vectors.iteritems()}
    products = {k: mpi_wrap_operator(v, with_apply2=with_apply2, array_type=array_type) if v else None
                for k, v in products.iteritems()} if products else None

    if use_with:
        d = mpi.get_object(obj_id)
        visualizer = MPIVisualizer(obj_id)
        return d.with_(operators=operators, functionals=functionals, vector_operators=vectors, products=products,
                       visualizer=visualizer, cache_region=None)
    else:
        return MPIDiscretization(obj_id, operators, functionals, vectors, products, array_type=array_type)


def _mpi_wrap_discretization_manage_operators(obj_id):
    d = mpi.get_object(obj_id)
    operators = {k: mpi.manage_object(v) if v else None for k, v in sorted(d.operators.iteritems())}
    functionals = {k: mpi.manage_object(v) if v else None for k, v in sorted(d.functionals.iteritems())}
    vectors = {k: mpi.manage_object(v) if v else None for k, v in sorted(d.vector_operators.iteritems())}
    products = {k: mpi.manage_object(v) if v else None for k, v in sorted(d.products.iteritems())} if d.products else None
    if mpi.rank0:
        return operators, functionals, vectors, products
