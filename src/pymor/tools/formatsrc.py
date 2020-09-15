# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from inspect import getsource

from pymor.core.config import is_jupyter


def format_source(obj):
    source = getsource(obj)

    if is_jupyter():
        from IPython.display import display, Code
        return Code(source, language='python')
    else:
        try:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import Terminal256Formatter
            return highlight(self.source, PythonLexer(), Terminal256Formatter())
        except ImportError:
            return source


def print_source(obj):
    source = format_source(obj)
    if is_jupyter():
        display(source)
    else:
        print(source)


def source_repr(obj):
    source = format_source(obj)
    if is_jupyter():
        display(source)
        return ''
    else:
        return source
