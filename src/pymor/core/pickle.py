# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains methods for object serialization.

Instead of importing serialization functions from Python's
:mod:`pickle` module directly, you should use the `dump`, `dumps`,
`load`, `loads` functions defined here. In particular, these
methods will use :func:`dumps_function` to serialize
function objects which cannot be pickled by Python's standard
methods. Note, however, pickling such methods should be avoided
since the implementation of :func:`dumps_function` uses non-portable
implementation details of CPython to achieve its goals.
"""

from __future__ import absolute_import


import marshal
import opcode
from types import FunctionType, ModuleType


try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


PicklingError = pickle.PicklingError
UnpicklingError = pickle.UnpicklingError
PROTOCOL = pickle.HIGHEST_PROTOCOL


# on CPython provide pickling methods which use
# dumps_function in case pickling of a function fails
import platform
if platform.python_implementation() == 'CPython':

    def dump(obj, file, protocol=None):
        pickler = pickle.Pickler(file, protocol=PROTOCOL)
        pickler.persistent_id = _function_pickling_handler
        pickler.dump(obj)


    def dumps(obj, protocol=None):
        file = StringIO()
        pickler = pickle.Pickler(file, protocol=PROTOCOL)
        pickler.persistent_id = _function_pickling_handler
        pickler.dump(obj)
        return file.getvalue()


    def load(file):
        unpickler = pickle.Unpickler(file)
        unpickler.persistent_load = _function_unpickling_handler
        return unpickler.load()


    def loads(str):
        file = StringIO(str)
        unpickler = pickle.Unpickler(file)
        unpickler.persistent_load = _function_unpickling_handler
        return unpickler.load()

else:
    from functools import partial
    dump = partial(pickle.dump, protocol=PROTOCOL)
    dumps = partial(pickle.dumps, protocol=PROTOCOL)
    load = pickle.load
    loads = pickle.loads


# The following method is a slightly modified version of
# a recipe from the "Python Cookbook", 3rd edition.

def _generate_opcode(code_object):
    HAVE_ARGUMENT = opcode.HAVE_ARGUMENT
    EXTENDED_ARG = opcode.EXTENDED_ARG

    codebytes = bytearray(code_object.co_code)
    extended_arg = 0
    i = 0
    n = len(codebytes)
    while i < n:
        op = codebytes[i]
        i += 1
        if op >= HAVE_ARGUMENT:
            oparg = codebytes[i] + codebytes[i + 1] * 256 + extended_arg
            extended_arg = 0
            i += 2
            if op == EXTENDED_ARG:
                extended_arg = oparg * 65536
                continue
        else:
            oparg = None
        yield (op, oparg)


def _global_names(code_object):
    '''Return all names in code_object.co_names which are used in a LOAD_GLOBAL statement.'''
    LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
    indices = set(i for o, i in _generate_opcode(code_object) if o == LOAD_GLOBAL)
    names = code_object.co_names
    return [names[i] for i in indices]


class Module(object):

    def __init__(self, mod):
        self.mod = mod

    def __getstate__(self):
        if not hasattr(self.mod, '__package__'):
            raise PicklingError
        return self.mod.__package__

    def __setstate__(self, s):
        self.mod = __import__(s)


def dumps_function(function):
    '''Tries hard to pickle a function object:

        1. The function's code object is serialized using the :mod:`marshal` module.
        2. For all global names used in the function's code object the corresponding
           object in the function's global namespace is pickled. In case this object
           is a module, the modules __package__ name is pickled.
        3. All default arguments are pickled.
        4. All objects in the function's closure are pickled.

    Note that also this is heavily implementation specific and will probably only
    work with CPython. If possible, avoid using this method.
    '''
    closure = None if function.func_closure is None else [c.cell_contents for c in function.func_closure]
    code = marshal.dumps(function.func_code)
    func_globals = function.func_globals
    func_dict = function.func_dict

    def wrap_modules(x):
        return Module(x) if isinstance(x, ModuleType) else x

    # note that global names in function.func_code can also refer to builtins ...
    globals_ = {k: wrap_modules(func_globals[k]) for k in _global_names(function.func_code) if k in func_globals}
    return dumps((function.func_name, code, globals_, function.func_defaults, closure, func_dict))


def loads_function(s):
    '''Restores a function serialized with :func:`dumps_function`.'''
    name, code, globals_, defaults, closure, func_dict = loads(s)
    code = marshal.loads(code)
    for k, v in globals_.iteritems():
        if isinstance(v, Module):
            globals_[k] = v.mod
    if closure is not None:
        import ctypes
        ctypes.pythonapi.PyCell_New.restype = ctypes.py_object
        ctypes.pythonapi.PyCell_New.argtypes = [ctypes.py_object]
        closure = tuple(ctypes.pythonapi.PyCell_New(c) for c in closure)
    globals_['__builtins__'] = __builtins__
    r = FunctionType(code, globals_, name, defaults, closure)
    r.func_dict = func_dict
    return r


def _function_pickling_handler(f):
    if f.__class__ is FunctionType:
        if f.__module__ != '__main__':
            try:
                return 'A' + pickle.dumps(f)
            except (TypeError, PicklingError):
                return 'B' + dumps_function(f)
        else:
            return 'B' + dumps_function(f)
    else:
        return None


def _function_unpickling_handler(persid):
    mode, data = persid[0], persid[1:]
    if mode == 'A':
        return pickle.loads(data)
    elif mode == 'B':
        return loads_function(data)
    else:
        return UnpicklingError
