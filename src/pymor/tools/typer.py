#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from enum import Enum


def Choices(choices):

    class StringEnum(str, Enum):
        pass

    return StringEnum('Choices', ((o, o) for o in choices.split(' ')))
