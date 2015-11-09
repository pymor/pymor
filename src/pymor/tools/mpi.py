# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" This module provides helper methods to use pyMOR in parallel with MPI.

Executing this module will execute :func:`event_loop` on all MPI ranks
except for rank 0 where either a given script is executed::

    mpirun -n 16 python -m pymor.tools.mpi /path/to/script

or an interactive session is started::

    mpirun -n 16 python -m pymor.tools.mpi

When IPython is available, an IPython kernel is started which
the user can connect to by calling::

    ipython console --existing file_name_printed_by_ipython.json

(Starting the IPython console directly will not work properly
with most MPI implementations.)
When IPython is not available, the builtin python REPL is started.

When :func:`event_loop` is running on the MPI ranks, :func:`call`
can be run on rank 0 to execute the same Python function (given
as first arguments) simultaneously on all MPI ranks (including
rank 0). Calling :func:`quit` will exit :func:`event_loop` on
all MPI ranks.

Moreover, this module provides several helper methods which are
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
return value and return the corresponding `ObjectId`. The functions
:func:`method_call` and :func:`method_call_manage` are given an
`ObjectId` and a string as first and second argument and execute
method named by the second argument on the object referred to by the
first argument.
"""

from __future__ import absolute_import, division, print_function

import sys

try:
    from mpi4py import MPI
    HAVE_MPI = True

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank0 = rank == 0
    size = comm.Get_size()
    parallel = (size > 1)
    finished = False

except ImportError:
    HAVE_MPI = False


_managed_objects = {}
_object_counter = 0


################################################################################


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
                break
            else:
                method(*args, **kwargs)
        except:
            import traceback
            print("Caught exception on MPI rank {}:".format(rank))
            traceback.print_exception(*sys.exc_info())


def call(method, *args, **kwargs):
    """Execute method on all MPI ranks.

    Assuming :func:`event_loop` is running on all MPI ranks
    (except rank 0), this will execute `method` on all
    ranks (including rank 0) with sequential arguments
    `args` and keyword arguments `kwargs`.

    Parameters
    ----------
    method
        The function to execute on all ranks. Must be picklable.
    args
        The sequential arguments for `method`.
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
    global finished
    comm.bcast(('QUIT', None, None))
    finished = True


################################################################################


def mpi_info():
    """Print some information on the MPI setup.

    Intended to be used in conjunction with :func:`call`.
    """
    data = comm.gather((rank, MPI.Get_processor_name()), root=0)
    if rank0:
        print('\n'.join('{}: {}'.format(rank, processor) for rank, processor in data))


def run_code(code):
    """Execute the code string `code`.

    Intended to be used in conjunction with :func:`call`.
    """
    exec code


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
             **{k: (get_object(v) if type(v) is ObjectId else v) for k, v in kwargs.iteritems()})


def function_call_manage(f, *args, **kwargs):
    """Execute the function `f` and manage the return value.

    Intended to be used in conjunction with :func:`call`.
    The return value of `f` is managed by calling :func:`manage_object`
    and the corresponding :class:`ObjectId` is returned.
    Arguments of type :class:`ObjectId` are transparently
    mapped to the object they refer to.
    """
    return manage_object(function_call(f, *args, **kwargs))


def method_call(obj_id, name, *args, **kwargs):
    """Execute a method with given arguments.

    Intended to be used in conjunction with :func:`call`.
    Arguments of type :class:`ObjectId` are transparently
    mapped to the object they refer to.

    Parameters
    ----------
    obj_id
        The :class:`ObjectId` of the object on which to call
        the method.
    name
        Name of the method to call.
    args
        Sequential arguments for the method.
    kwargs
        Keyword arguments for the method.
    """
    obj = get_object(obj_id)
    return getattr(obj, name)(*((get_object(arg) if type(arg) is ObjectId else arg) for arg in args),
                              **{k: (get_object(v) if type(v) is ObjectId else v) for k, v in kwargs.iteritems()})


def method_call_manage(obj_id, name, *args, **kwargs):
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
    name
        Name of the method to call.
    args
        Sequential arguments for the method.
    kwargs
        Keyword arguments for the method.
    """
    return manage_object(method_call(obj_id, name, *args, **kwargs))


################################################################################


class ObjectId(int):
    """A handle to an MPI distributed object."""
    pass


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
    """Return the object referred to by `obj_id` from the registry."""
    del _managed_objects[obj_id]


################################################################################


if __name__ == '__main__':
    assert HAVE_MPI
    if rank0:
        if len(sys.argv) >= 2:
            filename = sys.argv[1]
            sys.argv = sys.argv[:1] + sys.argv[2:]
            execfile(filename)
            import pymor.tools.mpi  # this is different from __main__
            pymor.tools.mpi.quit()  # change global state in the right module
        else:
            try:
                import IPython
                IPython.start_kernel()  # only start a kernel since mpirun messes up the terminal
            except ImportError:
                import code
                code.interact()
    else:
        event_loop()
