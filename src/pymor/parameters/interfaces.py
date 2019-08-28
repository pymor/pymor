# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

from pymor.core.interfaces import ImmutableInterface, abstractmethod
from pymor.parameters.base import Parametric


class ParameterSpaceInterface(ImmutableInterface):
    """Interface for |Parameter| spaces.

    Attributes
    ----------
    parameter_type
        |ParameterType| of the space.
    """

    parameter_type = None

    @abstractmethod
    def contains(self, mu):
        """`True` if `mu` is contained in the space."""
        pass


class ParameterFunctionalInterface(ImmutableInterface, Parametric):
    """Interface for |Parameter| functionals.

    A parameter functional is simply a function mapping a |Parameter| to
    a number.
    """

    @abstractmethod
    def evaluate(self, mu=None):
        """Evaluate the functional for the given |Parameter| `mu`."""
        pass

    @abstractmethod
    def d_mu(self, component, index=()):
        """Return the functionals's derivative with respect to an index of a parameter component.

        Parameters
        ----------
        component
            Parameter component
        index
            index in the parameter component

        Returns
        -------
        New |Parameter| functional representing the partial derivative.
        """
        pass

    def __call__(self, mu=None):
        return self.evaluate(mu)

    def __mul__(self, other):
        from pymor.parameters.functionals import ProductParameterFunctional
        if not isinstance(other, (Number, ParameterFunctionalInterface)):
            return NotImplemented
        return ProductParameterFunctional([self, other])

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1.)

    def _check_and_parse_input(self, component, index):
        # check whether component is in parameter_type
        if component not in self.parameter_type:
            return False, None
        # check whether index has the correct shape
        if isinstance(index, Number):
            index = (index,)
        index = tuple(index)
        for idx in index:
            assert isinstance(idx, Number)
        shape = self.parameter_type[component]
        for i,idx in enumerate(index):
            assert idx < shape[i], 'wrong input `index` given'
        return True, index
