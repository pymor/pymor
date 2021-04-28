# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from mypy.plugin import Plugin
from mypy.plugins.common import add_method_to_class, Argument, Var
from mypy.types import NoneType, AnyType, TypeOfAny


BasicObject_subclasses = {'pymor.core.base.BasicObject'}


def base_class_hook(ctx):
    BasicObject_subclasses.add(ctx.cls.fullname)
    any_type = AnyType(TypeOfAny.explicit)
    str_type = ctx.api.named_type('str')
    add_method_to_class(
        ctx.api, ctx.cls, '__auto_init',
        [Argument(Var('locals_'), ctx.api.named_type('dict', [str_type, any_type]), None, 0)],
        NoneType()
    )


class PymorPlugin(Plugin):
    def get_base_class_hook(self, fullname: str):
        if fullname in BasicObject_subclasses:
            # Add __auto_init to classes inheriting from BasicObject
            # Note: It's not clear to me if it is guaranteed that superclasses
            # are visited first, but so far this seems to work ...
            return base_class_hook


def plugin(version: str):
    return PymorPlugin
