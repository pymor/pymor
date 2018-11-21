# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import ImmutableInterface
from pymor.discretizations.basic import DiscretizationBase
from pymor.operators.mpi import mpi_wrap_operator
from pymor.tools import mpi
from pymor.vectorarrays.mpi import MPIVectorSpace, _register_local_space


class MPIDiscretization(DiscretizationBase):
    """Wrapper class for MPI distributed |Discretizations|.

    Given a single-rank implementation of a |Discretization|, this
    wrapper class uses the event loop from :mod:`pymor.tools.mpi`
    to allow an MPI distributed usage of the |Discretization|.
    The underlying implementation needs to be MPI aware.
    In particular, the discretization's
    :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`
    method has to perform an MPI parallel solve of the discretization.

    Note that this class is not intended to be instantiated directly.
    Instead, you should use :func:`mpi_wrap_discretization`.

    Parameters
    ----------
    obj_id
        :class:`~pymor.tools.mpi.ObjectId` of the local
        |Discretization| on each rank.
    operators
        Dictionary of all |Operators| contained in the discretization,
        wrapped for use on rank 0. Use :func:`mpi_wrap_discretization`
        to automatically wrap all operators of a given MPI-aware
        |Discretization|.
    products
        See `operators`.
    pickle_local_spaces
        See :class:`~pymor.operators.mpi.MPIOperator`.
    space_type
        See :class:`~pymor.operators.mpi.MPIOperator`.
    """

    def __init__(self, obj_id, operators, products=None,
                 pickle_local_spaces=True, space_type=MPIVectorSpace):
        d = mpi.get_object(obj_id)
        visualizer = MPIVisualizer(obj_id)
        super().__init__(operators=operators, products=products,
                         visualizer=visualizer, cache_region=None, name=d.name)
        self.obj_id = obj_id
        local_spaces = mpi.call(_MPIDiscretization_get_local_spaces, obj_id, pickle_local_spaces)
        if all(ls == local_spaces[0] for ls in local_spaces):
            local_spaces = (local_spaces[0],)
        self.solution_space = space_type(local_spaces)
        self.build_parameter_type(d)
        self.parameter_space = d.parameter_space

    def _solve(self, mu=None):
        return self.solution_space.make_array(
            mpi.call(mpi.method_call_manage, self.obj_id, 'solve', mu=mu)
        )

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


def _MPIDiscretization_get_local_spaces(self, pickle_local_spaces):
    self = mpi.get_object(self)
    local_space = self.solution_space
    if not pickle_local_spaces:
        local_space = _register_local_space(local_space)
    local_spaces = mpi.comm.gather(local_space, root=0)
    if mpi.rank0:
        return tuple(local_spaces)


class MPIVisualizer(ImmutableInterface):

    def __init__(self, d_obj_id):
        self.d_obj_id = d_obj_id

    def visualize(self, U, d, **kwargs):
        if isinstance(U, tuple):
            U = tuple(u.obj_id for u in U)
        else:
            U = U.obj_id
        mpi.call(_MPIVisualizer_visualize, self.d_obj_id, U, **kwargs)


def _MPIVisualizer_visualize(d, U, **kwargs):
    d = mpi.get_object(d)
    if isinstance(U, tuple):
        U = tuple(mpi.get_object(u) for u in U)
    else:
        U = mpi.get_object(U)
    d.visualize(U, **kwargs)


def mpi_wrap_discretization(local_discretizations, use_with=False, with_apply2=False,
                            pickle_local_spaces=True, space_type=MPIVectorSpace):
    """Wrap MPI distributed local |Discretizations| to a global |Discretization| on rank 0.

    Given MPI distributed local |Discretizations| referred to by the
    :class:`~pymor.tools.mpi.ObjectId` `local_discretizations`, return a new |Discretization|
    which manages these distributed discretizations from rank 0. This
    is done by first wrapping all |Operators| of the |Discretization| using
    :func:`~pymor.operators.mpi.mpi_wrap_operator`.

    Alternatively, `local_discretizations` can be a callable (with no arguments)
    which is then called on each rank to instantiate the local |Discretizations|.

    When `use_with` is `False`, an :class:`MPIDiscretization` is instantiated
    with the wrapped operators. A call to
    :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`
    will then use an MPI parallel call to the
    :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`
    methods of the wrapped local |Discretizations| to obtain the solution.
    This is usually what you want when the actual solve is performed by
    an implementation in the external solver.

    When `use_with` is `True`, :meth:`~pymor.core.interfaces.ImmutableInterface.with_`
    is called on the local |Discretization| on rank 0, to obtain a new
    |Discretization| with the wrapped MPI |Operators|. This is mainly useful
    when the local discretizations are generic |Discretizations| as in
    :mod:`pymor.discretizations.basic` and
    :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`
    is implemented directly in pyMOR via operations on the contained
    |Operators|.

    Parameters
    ----------
    local_discretizations
        :class:`~pymor.tools.mpi.ObjectId` of the local |Discretizations|
        on each rank or a callable generating the |Discretizations|.
    use_with
        See above.
    with_apply2
        See :class:`~pymor.operators.mpi.MPIOperator`.
    pickle_local_spaces
        See :class:`~pymor.operators.mpi.MPIOperator`.
    space_type
        See :class:`~pymor.operators.mpi.MPIOperator`.
    """

    if not isinstance(local_discretizations, mpi.ObjectId):
        local_discretizations = mpi.call(mpi.function_call_manage, local_discretizations)

    operators, products = mpi.call(_mpi_wrap_discretization_manage_operators, local_discretizations)

    operators = {k: mpi_wrap_operator(v, with_apply2=with_apply2,
                                      pickle_local_spaces=pickle_local_spaces, space_type=space_type) if v else None
                 for k, v in operators.items()}
    products = {k: mpi_wrap_operator(v, with_apply2=with_apply2,
                                     pickle_local_spaces=pickle_local_spaces, space_type=space_type) if v else None
                for k, v in products.items()}

    if use_with:
        d = mpi.get_object(local_discretizations)
        visualizer = MPIVisualizer(local_discretizations)
        return d.with_(operators=operators, products=products, visualizer=visualizer, cache_region=None)
    else:
        return MPIDiscretization(local_discretizations, operators, products,
                                 pickle_local_spaces=pickle_local_spaces, space_type=space_type)


def _mpi_wrap_discretization_manage_operators(obj_id):
    d = mpi.get_object(obj_id)
    operators = {k: mpi.manage_object(v) if v else None for k, v in sorted(d.operators.items())}
    products = {k: mpi.manage_object(v) if v else None for k, v in sorted(d.products.items())} if d.products else {}
    if mpi.rank0:
        return operators, products
