# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import sys

try:
    from mpi4py import MPI
    HAVE_MPI = True

    comm = MPI.COMM_WORLD
    rank = rank = comm.Get_rank()
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
            print("Caught exception on MPI rank {}:".format(rank), sys.exc_info()[0])


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


def manage_object(obj):
    global _object_counter
    obj_id = _object_counter
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


def method_call(obj_id, name, *args, **kwargs):
    obj = get_object(obj_id)
    return getattr(obj, name)(*args, **kwargs)


def method_call_manage(obj_id, name, *args, **kwargs):
    obj = get_object(obj_id)
    return manage_object(getattr(obj, name)(*args, **kwargs))


def method_call1(obj_id, name, obj_id1, *args, **kwargs):
    obj = get_object(obj_id)
    obj1 = get_object(obj_id1)
    return getattr(obj, name)(obj1, *args, **kwargs)


def method_call1_manage(obj_id, name, obj_id1, *args, **kwargs):
    obj = get_object(obj_id)
    obj1 = get_object(obj_id1)
    return manage_object(getattr(obj, name)(obj1, *args, **kwargs))


def method_call2(obj_id, name, obj_id1, obj_id2, *args, **kwargs):
    obj = get_object(obj_id)
    obj1 = get_object(obj_id1)
    obj2 = get_object(obj_id2)
    return getattr(obj, name)(obj1, obj2, *args, **kwargs)


def method_call2_manage(obj_id, name, obj_id1, obj_id2, *args, **kwargs):
    obj = get_object(obj_id)
    obj1 = get_object(obj_id1)
    obj2 = get_object(obj_id2)
    return manage_object(getattr(obj, name)(obj1, obj2, *args, **kwargs))


def run_code(code):
    exec code


def import_module(path):
    __import__(path)


if __name__ == '__main__':
    assert HAVE_MPI
    if rank0:
        import sys
        assert 1 <= len(sys.argv) <= 2
        if len(sys.argv) == 2:
            execfile(sys.argv[1])
        else:
            try:
                import IPython
                IPython.start_kernel()  # only start a kernel since mpirun messes up the terminal
            except ImportError:
                import code
                code.interact()
    else:
        event_loop()
