# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.interfaces import OperatorInterface


class IOSystem(DiscretizationInterface):
    """Base class for input-output systems

    Attributes
    ----------
    cont_time
        `True` if the system is continuous-time, otherwise discrete-time.
    m
        Number of inputs.
    p
        Number of outputs.
    """
    cont_time = None
    m = None
    p = None


class LTISystem(IOSystem):
    """Class for linear time-invariant systems

    This class describes input-state-output systems given by::

        E x'(t) = A x(t) + B u(t)
           y(t) = C x(t) + D u(t)

    if continuous-time, or::

        E x(k + 1) = A x(k) + B u(k)
          y(k)     = C x(k) + D u(k)

    if discrete-time, where A, B, C, D, and E are linear operators.

    Parameters
    ----------
    A
        The |Operator| A.
    B
        The |Operator| B.
    C
        The |Operator| C.
    D
        The |Operator| D or None (then D is assumed to be zero).
    E
        The |Operator| E or None (then E is assumed to be the identity).
    cont_time
        `True` if the system is continuous-time, otherwise discrete-time.

    Attributes
    ----------
    n
        Size of x.
    """
    linear = True

    def __init__(self, A, B, C, D=None, E=None, cont_time=True):
        assert isinstance(A, OperatorInterface) and A.linear
        assert isinstance(B, OperatorInterface) and B.linear
        assert isinstance(C, OperatorInterface) and C.linear
        assert A.source == A.range == B.range == C.source
        assert (D is None or isinstance(D, OperatorInterface) and D.linear and D.source == B.source and
                D.range == C.range)
        assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
        assert cont_time in {True, False}

        self.n = A.source.dim
        self.m = B.source.dim
        self.p = C.range.dim
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.cont_time = cont_time

    def _solve(self):
        raise NotImplementedError('Discretization has no solver.')

    def norm(self):
        if self.cont_time:
            import pycmess
            import numpy as np
            from pymor.vectorarrays.numpy import NumpyVectorArray
            import numpy.linalg as npla

            Z = pycmess.lyap(self.A._matrix, self.E._matrix, self.B._matrix)
            Z = np.array(Z)
            Z = NumpyVectorArray(Z)
            return npla.norm(self.C.apply(Z).data)
        else:
            raise NotImplementedError
