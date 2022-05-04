# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.block import BlockOperator
from pymor.operators.constructions import IdentityOperator
from pymor.vectorarrays.block import BlockVectorSpace


class CanonicalSymplecticFormOperator(BlockOperator):
    """|Operator| representing a canonically symplectic form.

    Parameters
    ----------
    phase_space
        The phase space of a |SymplecticBasis|.
    """

    def __init__(self, phase_space):
        assert (isinstance(phase_space, BlockVectorSpace)
                and len(phase_space.subspaces) == 2
                and phase_space.subspaces[0] == phase_space.subspaces[1])
        self.__auto_init(locals())
        half_space = phase_space.subspaces[0]
        super().__init__([[None, IdentityOperator(half_space)],
                          [-IdentityOperator(half_space), None]])
