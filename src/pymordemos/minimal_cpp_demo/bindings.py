# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import pybindgen


# ReturnValue for converting a raw C pointer to a PyBuffer object
# from pybindgen/examples/buffer/modulegen.py
class BufferReturn(pybindgen.ReturnValue):
    CTYPES = []

    def __init__(self, ctype, length_expression):
        super(BufferReturn, self).__init__(ctype, is_const=False)
        self.length_expression = length_expression

    def convert_c_to_python(self, wrapper):
        pybuf = wrapper.after_call.declare_variable("PyObject*", "pybuf")
        wrapper.after_call.write_code("%s = PyBuffer_FromReadWriteMemory(retval, %s);"
                                      % (pybuf, self.length_expression))
        wrapper.build_params.add_parameter("N", [pybuf], prepend=True)


mod = pybindgen.Module('discretization')
mod.add_include('"discretization.hh"')

mod.add_container('std::vector<double>', 'double', 'vector')  # declare a container only once

Vector = mod.add_class('Vector')
Vector.add_constructor([pybindgen.param('int', 'size'), pybindgen.param('double', 'value')])
Vector.add_constructor([pybindgen.param('const Vector&', 'othter')])

Vector.add_instance_attribute('dim', 'int', is_const=True)
Vector.add_method('scal', None, [pybindgen.param('double', 'val')])
Vector.add_method('axpy', None, [pybindgen.param('double', 'a'), pybindgen.param('const Vector&', 'x')])
Vector.add_method('dot', pybindgen.retval('double'), [pybindgen.param('const Vector&', 'other')], is_const=True)
Vector.add_method('almost_equal', pybindgen.retval('bool'),
                  [pybindgen.param('const Vector&', 'other'),
                   pybindgen.param('double', 'rtol'),
                   pybindgen.param('double', 'atol')], is_const=True)
Vector.add_method('data', BufferReturn('double*', 'self->obj->dim * sizeof(double)'), [])

DiffusionOperator = mod.add_class('DiffusionOperator')
DiffusionOperator.add_constructor([pybindgen.param('int', 'n'),
                                   pybindgen.param('double', 'left'),
                                   pybindgen.param('double', 'right')])
DiffusionOperator.add_instance_attribute('dim_source', 'int', is_const=True)
DiffusionOperator.add_instance_attribute('dim_range', 'int', is_const=True)
DiffusionOperator.add_method('apply', None, [pybindgen.param('const Vector&', 'U'),
                                             pybindgen.param('Vector&', 'V')], is_const=True)

mod.generate(pybindgen.FileCodeSink(open("bindings_generated.cpp", 'w')))
