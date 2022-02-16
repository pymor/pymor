# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import inspect
import textwrap

import numpy as np

from pymor.core.defaults import defaults


def register_format_handler(cls, handler):
    _format_handlers[cls] = handler


def _format_generic(obj, max_width, verbosity, override={}):
    init_sig = inspect.signature(obj.__init__)
    keys, vals = [], []
    for arg, description in init_sig.parameters.items():
        if verbosity < 2 and description.default == description.empty:
            key = ''
        else:
            key = f'{arg}='
        if arg == 'self':
            continue
        if arg in override:
            val = override[arg]
            if val is None:
                continue
            vals.append(override[arg])
        else:
            val = getattr(obj, arg, '??')
            try:
                if verbosity < 3 and val == description.default:
                    continue
            except (ValueError, TypeError):  # comparison of numpy arrays, NGSolve FESpaces
                pass
            vals.append(_recurse(val, max_width - len(key) - 4, verbosity))
        keys.append(key)

    if verbosity > 0 and (sum(len(k) + len(v) + 2 for k, v in zip(keys, vals)) + len(type(obj).__name__) > max_width
                          or any('\n' in v for v in vals)):
        args = [f'    {k}{indent_value(v, len(k) + 4)}' for k, v in zip(keys, vals)]
        args = ",\n".join(args)
        return f'''{type(obj).__name__}(
{args})'''
    else:
        args = [f'{k}{v}' for k, v in zip(keys, vals)]
        return f'{type(obj).__name__}({", ".join(args)})'


def _format_list_tuple(val, max_width, verbosity):
    brackets = '()' if type(val) is tuple else '[]'
    reprs = [repr(v) for v in val]
    if verbosity > 0 and (any('\n' in r for r in reprs) or sum(len(r) + 2 for r in reprs) + 2 > max_width):
        reprs = ',\n '.join(indent_value(r, 1) for r in reprs)
        return brackets[0] + reprs + brackets[1]
    else:
        return brackets[0] + ', '.join(reprs) + brackets[1]


def _format_dict(val, max_width, verbosity):
    if not val:
        return '{}'
    keys, vals = zip(*val.items())
    reprs = [repr(v) for v in vals]
    if verbosity > 0 and (any('\n' in r for r in reprs)
                          or sum(len(k) + len(r) + 4 for k, r in zip(keys, reprs)) + 2 > max_width):
        reprs = ',\n '.join(indent_value(f'{k}: {r}', 1) for k, r in zip(keys, reprs))
        return '{' + reprs + '}'
    else:
        return '{' + ', '.join(f'{k}: {r}' for k, r in zip(keys, reprs)) + '}'


def _format_array(val, max_width, verbosity):
    r = repr(val)
    if len(r) <= max_width * 3 and '\n' in r:
        r_one_line = ' '.join(rr.strip() for rr in r.split('\n'))
        if len(r_one_line) <= max_width:
            return r_one_line
        else:
            return r
    else:
        return r


_format_handlers = {}
register_format_handler(list, _format_list_tuple)
register_format_handler(tuple, _format_list_tuple)
register_format_handler(dict, _format_dict)
register_format_handler(np.ndarray, _format_array)


def _recurse(obj, max_width, verbosity):
    if hasattr(obj, '_format_repr'):
        return obj._format_repr(max_width, verbosity)

    handler = None
    for cls in type(obj).__mro__:
        try:
            handler = _format_handlers[cls]
        except KeyError:
            pass

    if handler:
        return handler(obj, max_width, verbosity)
    else:
        return repr(obj)


@defaults('max_width', 'verbosity')
def format_repr(obj, max_width=120, verbosity=1):
    return _recurse(obj, max_width, verbosity)


def indent_value(val, indent):
    if '\n' in val:
        return textwrap.indent(val, ' ' * indent)[indent:]
    else:
        return val
