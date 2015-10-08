# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

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


def event_loop():
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


def quit():
    global finished
    comm.bcast(('QUIT', None, None))
    finished = True


def call(method, *args, **kwargs):
    assert rank0
    if finished:
        return
    comm.bcast((method, args, kwargs), root=0)
    return method(*args, **kwargs)


class ObjectId(int):
    pass


def manage_object(obj):
    global _object_counter
    obj_id = ObjectId(_object_counter)
    _managed_objects[obj_id] = obj
    _object_counter += 1
    return obj_id


def get_object(obj_id):
    return _managed_objects[obj_id]


def remove_object(obj_id):
    del _managed_objects[obj_id]


def mpi_info():
    data = comm.gather((rank, MPI.Get_processor_name()), root=0)
    if rank0:
        print('\n'.join('{}: {}'.format(rank, processor) for rank, processor in data))


def function_call(f, *args, **kwargs):
    return f(*((get_object(arg) if type(arg) is ObjectId else arg) for arg in args),
             **{k: (get_object(v) if type(v) is ObjectId else v) for k, v in kwargs.iteritems()})


def function_call_manage(f, *args, **kwargs):
    return manage_object(function_call(f, *args, **kwargs))


def method_call(obj_id, name, *args, **kwargs):
    obj = get_object(obj_id)
    return getattr(obj, name)(*((get_object(arg) if type(arg) is ObjectId else arg) for arg in args),
                              **{k: (get_object(v) if type(v) is ObjectId else v) for k, v in kwargs.iteritems()})


def method_call_manage(obj_id, name, *args, **kwargs):
    return manage_object(method_call(obj_id, name, *args, **kwargs))


def run_code(code):
    exec code


def import_module(path):
    __import__(path)


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
