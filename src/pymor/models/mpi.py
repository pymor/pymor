# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import ImmutableInterface
from pymor.models.basic import ModelBase
from pymor.operators.mpi import mpi_wrap_operator
from pymor.tools import mpi
from pymor.vectorarrays.mpi import MPIVectorSpace, _register_local_space


class MPIModel(ModelBase):
    """Wrapper class for MPI distributed |Models|.

    Given a single-rank implementation of a |Model|, this
    wrapper class uses the event loop from :mod:`pymor.tools.mpi`
    to allow an MPI distributed usage of the |Model|.
    The underlying implementation needs to be MPI aware.
    In particular, the model's
    :meth:`~pymor.models.interfaces.ModelInterface.solve`
    method has to perform an MPI parallel solve of the model.

    Note that this class is not intended to be instantiated directly.
    Instead, you should use :func:`mpi_wrap_model`.

    Parameters
    ----------
    obj_id
        :class:`~pymor.tools.mpi.ObjectId` of the local
        |Model| on each rank.
    operators
        Dictionary of all |Operators| contained in the model,
        wrapped for use on rank 0. Use :func:`mpi_wrap_model`
        to automatically wrap all operators of a given MPI-aware
        |Model|.
    products
        See `operators`.
    pickle_local_spaces
        See :class:`~pymor.operators.mpi.MPIOperator`.
    space_type
        See :class:`~pymor.operators.mpi.MPIOperator`.
    """

    def __init__(self, obj_id, operators, products=None,
                 pickle_local_spaces=True, space_type=MPIVectorSpace):
        m = mpi.get_object(obj_id)
        visualizer = MPIVisualizer(obj_id)
        super().__init__(operators=operators, products=products,
                         visualizer=visualizer, cache_region=None, name=m.name)
        self.obj_id = obj_id
        local_spaces = mpi.call(_MPIModel_get_local_spaces, obj_id, pickle_local_spaces)
        if all(ls == local_spaces[0] for ls in local_spaces):
            local_spaces = (local_spaces[0],)
        self.solution_space = space_type(local_spaces)
        self.build_parameter_type(m)
        self.parameter_space = m.parameter_space

    def _solve(self, mu=None):
        return self.solution_space.make_array(
            mpi.call(mpi.method_call_manage, self.obj_id, 'solve', mu=mu)
        )

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


def _MPIModel_get_local_spaces(self, pickle_local_spaces):
    self = mpi.get_object(self)
    local_space = self.solution_space
    if not pickle_local_spaces:
        local_space = _register_local_space(local_space)
    local_spaces = mpi.comm.gather(local_space, root=0)
    if mpi.rank0:
        return tuple(local_spaces)


class MPIVisualizer(ImmutableInterface):

    def __init__(self, m_obj_id):
        self.m_obj_id = m_obj_id

    def visualize(self, U, m, **kwargs):
        if isinstance(U, tuple):
            U = tuple(u.obj_id for u in U)
        else:
            U = U.obj_id
        mpi.call(_MPIVisualizer_visualize, self.m_obj_id, U, **kwargs)


def _MPIVisualizer_visualize(m, U, **kwargs):
    m = mpi.get_object(m)
    if isinstance(U, tuple):
        U = tuple(mpi.get_object(u) for u in U)
    else:
        U = mpi.get_object(U)
    m.visualize(U, **kwargs)


def mpi_wrap_model(local_models, use_with=False, with_apply2=False,
                   pickle_local_spaces=True, space_type=MPIVectorSpace):
    """Wrap MPI distributed local |Models| to a global |Model| on rank 0.

    Given MPI distributed local |Models| referred to by the
    :class:`~pymor.tools.mpi.ObjectId` `local_models`, return a new |Model|
    which manages these distributed models from rank 0. This
    is done by first wrapping all |Operators| of the |Model| using
    :func:`~pymor.operators.mpi.mpi_wrap_operator`.

    Alternatively, `local_models` can be a callable (with no arguments)
    which is then called on each rank to instantiate the local |Models|.

    When `use_with` is `False`, an :class:`MPIModel` is instantiated
    with the wrapped operators. A call to
    :meth:`~pymor.models.interfaces.ModelInterface.solve`
    will then use an MPI parallel call to the
    :meth:`~pymor.models.interfaces.ModelInterface.solve`
    methods of the wrapped local |Models| to obtain the solution.
    This is usually what you want when the actual solve is performed by
    an implementation in the external solver.

    When `use_with` is `True`, :meth:`~pymor.core.interfaces.ImmutableInterface.with_`
    is called on the local |Model| on rank 0, to obtain a new
    |Model| with the wrapped MPI |Operators|. This is mainly useful
    when the local models are generic |Models| as in
    :mod:`pymor.models.basic` and
    :meth:`~pymor.models.interfaces.ModelInterface.solve`
    is implemented directly in pyMOR via operations on the contained
    |Operators|.

    Parameters
    ----------
    local_models
        :class:`~pymor.tools.mpi.ObjectId` of the local |Models|
        on each rank or a callable generating the |Models|.
    use_with
        See above.
    with_apply2
        See :class:`~pymor.operators.mpi.MPIOperator`.
    pickle_local_spaces
        See :class:`~pymor.operators.mpi.MPIOperator`.
    space_type
        See :class:`~pymor.operators.mpi.MPIOperator`.
    """

    if not isinstance(local_models, mpi.ObjectId):
        local_models = mpi.call(mpi.function_call_manage, local_models)

    operators, products = mpi.call(_mpi_wrap_model_manage_operators, local_models)

    operators = {k: mpi_wrap_operator(v, with_apply2=with_apply2,
                                      pickle_local_spaces=pickle_local_spaces, space_type=space_type) if v else None
                 for k, v in operators.items()}
    products = {k: mpi_wrap_operator(v, with_apply2=with_apply2,
                                     pickle_local_spaces=pickle_local_spaces, space_type=space_type) if v else None
                for k, v in products.items()}

    if use_with:
        m = mpi.get_object(local_models)
        visualizer = MPIVisualizer(local_models)
        return m.with_(operators=operators, products=products, visualizer=visualizer, cache_region=None)
    else:
        return MPIModel(local_models, operators, products,
                        pickle_local_spaces=pickle_local_spaces, space_type=space_type)


def _mpi_wrap_model_manage_operators(obj_id):
    m = mpi.get_object(obj_id)
    operators = {k: mpi.manage_object(v) if v else None for k, v in sorted(m.operators.items())}
    products = {k: mpi.manage_object(v) if v else None for k, v in sorted(m.products.items())} if m.products else {}
    if mpi.rank0:
        return operators, products
