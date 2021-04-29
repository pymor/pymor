# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import cast

from mypy.plugin import Plugin, MethodContext, ClassDefContext
from mypy.nodes import Decorator, CallExpr, StrExpr, NameExpr
from mypy.plugins.common import add_method_to_class, Argument, Var
from mypy.types import NoneType, AnyType, TypeOfAny, UnionType, CallableType


BasicObject_subclasses = {'pymor.core.base.BasicObject'}


def add_auto_init(ctx: ClassDefContext):
    BasicObject_subclasses.add(ctx.cls.fullname)
    any_type = AnyType(TypeOfAny.explicit)
    str_type = ctx.api.named_type('str')
    add_method_to_class(
        ctx.api, ctx.cls, '__auto_init',
        [Argument(Var('locals_'), ctx.api.named_type('dict', [str_type, any_type]), None, 0)],
        NoneType()
    )


def fix_defaults(ctx: MethodContext):
    # the function we are decorating with @defaults
    decorated_function = ctx.arg_types[0][0]
    assert isinstance(decorated_function, CallableType)

    # find @defaults in the list of applied decorators
    assert isinstance(ctx.context, Decorator)
    decorator = None
    for dec in ctx.context.decorators:
        if isinstance(dec, CallExpr) and isinstance(dec.callee, NameExpr) and dec.callee.name == 'defaults':
            decorator = dec
            break
    assert decorator is not None

    # make types of arguments with defaults Optional[...]
    args_with_default = [cast(StrExpr, arg).value for arg in decorator.args]
    arg_types = [UnionType([arg, NoneType()]) if name in args_with_default else arg
                 for arg, name in zip(decorated_function.arg_types, decorated_function.arg_names)]

    return decorated_function.copy_modified(arg_types=arg_types)


class PymorPlugin(Plugin):
    def get_base_class_hook(self, fullname: str):
        if fullname in BasicObject_subclasses:
            # Add __auto_init to classes inheriting from BasicObject
            # Note: It's not clear to me if it is guaranteed that superclasses
            # are visited first, but so far this seems to work ...
            return add_auto_init

    def get_method_hook(self, fullname: str):
        if fullname == 'pymor.core.defaults.defaults.__call__':
            return fix_defaults


def plugin(version: str):
    return PymorPlugin
