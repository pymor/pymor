# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.block import BlockOperator
from pymor.operators.constructions import IdentityOperator


class CanonicalSymplecticFormOperator(BlockOperator):
    def __init__(self, half_space):
        """An |Operator| for a canonically symplectic form.
        """
        self.__auto_init(locals())
        super().__init__([[None, IdentityOperator(half_space)],
                          [-IdentityOperator(half_space), None]])
