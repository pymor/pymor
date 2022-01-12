# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from inspect import getsource

from pymor.core.config import is_jupyter


def format_source(obj):
    """Format source code of an object.

    Parameters
    ----------
    obj
        The object of which to format the source code.

    Returns
    -------
    source
        The source code as a string, possibly highlighted when used in a
        terminal.
    """
    source = getsource(obj)
    if is_jupyter():
        return source
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import Terminal256Formatter
        return highlight(source, PythonLexer(), Terminal256Formatter())
    except ImportError:
        return source


def print_source(obj):
    """Print source code of an object.

    Parameters
    ----------
    obj
        The object of which to print the source code.
    """
    source = format_source(obj)
    if is_jupyter():
        from IPython.display import Code, display
        display(Code(source, language='python'))
    else:
        print(source)
