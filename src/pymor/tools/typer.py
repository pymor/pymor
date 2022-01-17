#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from enum import Enum


def Choices(choices):
    """Multipe-choice options for typer.

    This is a convenicence function that creates string Enums to be
    used as the type of command-line arguments that can take a fixed set
    of values. For example, the command::

        @app.command()
        def main(arg: Choices('value1 value2')):
            pass

    takes one argument that may either have the value `value1` or the
    value `value2`.
    """

    class StringEnum(str, Enum):
        pass

    return StringEnum('Choices', ((o, o) for o in choices.split(' ')))
