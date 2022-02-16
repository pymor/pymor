# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module provides helper methods to use pyMOR in parallel with MPI.

Executing this module will run :func:`event_loop` on all MPI ranks
except for rank 0 where either a given script is executed::

    mpirun -n 16 python -m pymor.tools.mpi /path/to/script

or an interactive session is started::

    mpirun -n 16 python -m pymor.tools.mpi

When IPython is available, an IPython kernel is started which
the user can connect to by calling::

    ipython console --existing file_name_printed_by_ipython.json

(Starting the IPython console directly will not work properly
with most MPI implementations.)
When IPython is not available, the builtin Python REPL is started.

When :func:`event_loop` is running on the MPI ranks, :func:`call`
can be used on rank 0 to execute the same Python function (given
as first argument) simultaneously on all MPI ranks (including
rank 0). Calling :func:`quit` will exit :func:`event_loop` on
all MPI ranks.

Additionally, this module provides several helper methods which are
intended to be used in conjunction with :func:`call`: :func:`mpi_info`
will print a summary of all active MPI ranks, :func:`run_code`
will execute the given code string on all MPI ranks,
:func:`import_module` imports the module with the given path.

A simple object management is implemented with the
:func:`manage_object`, :func:`get_object` and :func:`remove_object`
methods. It is the user's responsibility to ensure that calls to
:func:`manage_object` are executed in the same order on all MPI ranks
to ensure that the returned :class:`ObjectId` refers to the same
distributed object on all ranks. The functions :func:`function_call`,
:func:`function_call_manage`, :func:`method_call`,
:func:`method_call_manage` map instances :class:`ObjectId`
transparently to distributed objects. :func:`function_call_manage` and
:func:`method_call_manage` will call :func:`manage_object` on the
return value and return the corresponding :class:`ObjectId`. The functions
:func:`method_call` and :func:`method_call_manage` are given an
:class:`ObjectId` and a string as first and second argument and execute
the method named by the second argument on the object referred to by the
first argument.
"""

import sys

from packaging.version import Version

from pymor.core.config import config
from pymor.core.defaults import defaults

if config.HAVE_MPI:
    import pymor.core.pickle
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    finished = False
    MPI.pickle.__init__(pymor.core.pickle.dumps, pymor.core.pickle.loads, pymor.core.pickle.PROTOCOL)
else:
    mpi4py_version = Version('0.0')
    rank = 0
    size = 1
    finished = True

rank0 = rank == 0
parallel = (size > 1)

_managed_objects = {}
_object_counter = 0
_event_loop_running = False


################################################################################


@defaults('auto_launch')
def event_loop_settings(auto_launch=True):
    """Settings for pyMOR's MPI event loop.

    Parameters
    ----------
    auto_launch
        If `True`, automatically execute :func:`event_loop` on
        all MPI ranks (except 0) when pyMOR is imported.
    """
    return {'auto_launch': auto_launch}


def launch_event_loop():
    global _event_loop_running
    if rank0:
        from pymor.core import defaults
        if defaults.defaults_changes() > 0:
            call(defaults.set_defaults, defaults.get_defaults(user=True, file=True, code=False))
        _event_loop_running = True
    else:
        event_loop()


def event_loop():
    """Launches an MPI-based event loop.

    Events can be sent by either calling :func:`call` on
    rank 0 to execute an arbitrary method on all ranks or
    by calling :func:`quit` to exit the loop.
    """
    assert not rank0
    while True:
        try:
            method, args, kwargs = comm.bcast(None)
            if method == 'QUIT':
                assert not _managed_objects
                break
            else:
                method(*args, **kwargs)
        except BaseException:
            import traceback
            print(f"Caught exception on MPI rank {rank}:")
            traceback.print_exception(*sys.exc_info())


def call(method, *args, **kwargs):
    """Execute method on all MPI ranks.

    Assuming :func:`event_loop` is running on all MPI ranks
    (except rank 0), this will execute `method` on all
    ranks (including rank 0) with positional arguments
    `args` and keyword arguments `kwargs`.

    Parameters
    ----------
    method
        The function to execute on all ranks (must be picklable).
    args
        The positional arguments for `method`.
    kwargs
        The keyword arguments for `method`.

    Returns
    -------
    The return value of `method` on rank 0.
    """
    assert rank0
    if finished:
        return
    comm.bcast((method, args, kwargs), root=0)
    return method(*args, **kwargs)


def quit():
    """Exit the event loop on all MPI ranks.

    This will cause :func:`event_loop` to terminate on all
    MPI ranks.
    """
    global finished, _event_loop_running
    if _managed_objects:
        from warnings import warn
        warn('Leaving MPI event loop while not all managed objects have been removed. '
             'This might be caused by a resource leak.')
        for obj_id in list(_managed_objects):
            call(remove_object, obj_id)
    comm.bcast(('QUIT', None, None))
    finished = True
    _event_loop_running = False


################################################################################


def mpi_info():
    """Print some information on the MPI setup.

    Intended to be used in conjunction with :func:`call`.
    """
    data = comm.gather((rank, MPI.Get_processor_name()), root=0)
    if rank0:
        print('\n'.join(f'{rank}: {processor}' for rank, processor in data))


def run_code(code):
    """Execute the code string `code`.

    Intended to be used in conjunction with :func:`call`.
    """
    exec(code)


def import_module(path):
    """Import the module named by `path`.

    Intended to be used in conjunction with :func:`call`.
    """
    __import__(path)


def function_call(f, *args, **kwargs):
    """Execute the function `f` with given arguments.

    Intended to be used in conjunction with :func:`call`.
    Arguments of type :class:`ObjectId` are transparently
    mapped to the object they refer to.
    """
    return f(*((get_object(arg) if type(arg) is ObjectId else arg) for arg in args),
             **{k: (get_object(v) if type(v) is ObjectId else v) for k, v in kwargs.items()})


def function_call_manage(f, *args, **kwargs):
    """Execute the function `f` and manage the return value.

    Intended to be used in conjunction with :func:`call`.
    The return value of `f` is managed by calling :func:`manage_object`
    and the corresponding :class:`ObjectId` is returned.
    Arguments of type :class:`ObjectId` are transparently
    mapped to the object they refer to.
    """
    return manage_object(function_call(f, *args, **kwargs))


def method_call(obj_id, name_, *args, **kwargs):
    """Execute a method with given arguments.

    Intended to be used in conjunction with :func:`call`.
    Arguments of type :class:`ObjectId` are transparently
    mapped to the object they refer to.

    Parameters
    ----------
    obj_id
        The :class:`ObjectId` of the object on which to call
        the method.
    `name_`
        Name of the method to call.
    args
        Positional arguments for the method.
    kwargs
        Keyword arguments for the method.
    """
    obj = get_object(obj_id)
    return getattr(obj, name_)(*((get_object(arg) if type(arg) is ObjectId else arg) for arg in args),
                               **{k: (get_object(v) if type(v) is ObjectId else v) for k, v in kwargs.items()})


def method_call_manage(obj_id, name_, *args, **kwargs):
    """Execute a method with given arguments and manage the return value.

    Intended to be used in conjunction with :func:`call`.
    The return value of the called method is managed by calling
    :func:`manage_object` and the corresponding :class:`ObjectId`
    is returned.  Arguments of type :class:`ObjectId` are transparently
    mapped to the object they refer to.

    Parameters
    ----------
    obj_id
        The :class:`ObjectId` of the object on which to call
        the method.
    `name_`
        Name of the method to call.
    args
        Positional arguments for the method.
    kwargs
        Keyword arguments for the method.
    """
    return manage_object(method_call(obj_id, name_, *args, **kwargs))


################################################################################


class ObjectId(int):
    """A handle to an MPI distributed object."""


def manage_object(obj):
    """Keep track of `obj` and return an :class:`ObjectId` handle."""
    global _object_counter
    obj_id = ObjectId(_object_counter)
    _managed_objects[obj_id] = obj
    _object_counter += 1
    return obj_id


def get_object(obj_id):
    """Return the object referred to by `obj_id`."""
    return _managed_objects[obj_id]


def remove_object(obj_id):
    """Remove the object referred to by `obj_id` from the registry."""
    del _managed_objects[obj_id]


################################################################################


if __name__ == '__main__':
    assert config.HAVE_MPI
    launch_event_loop()
    if rank0:
        if len(sys.argv) >= 2:
            filename = sys.argv[1]
            sys.argv = sys.argv[:1] + sys.argv[2:]
            exec(compile(open(filename, 'rt').read(), filename, 'exec'))
            import pymor.tools.mpi  # this is different from __main__
            pymor.tools.mpi.quit()  # change global state in the right module
        else:
            try:
                import IPython
                IPython.start_kernel()  # only start a kernel since mpirun messes up the terminal
            except ImportError:
                import code
                code.interact()
