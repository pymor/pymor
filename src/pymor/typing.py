# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Union, List, TYPE_CHECKING

from numpy import ndarray, integer, number, floating

if TYPE_CHECKING:
    from pymor.parameters.functionals import ParameterFunctional

# admissable indices VectorArray.__getitem__
SCALAR_INDICES = (int, integer)  # use with isinstance to check for a scalar index
ScalarIndex = Union[int, integer]
Index = Union[int, integer, slice, List[Union[int, integer]], ndarray]


Real = Union[float, floating]
RealOrComplex = Union[float, complex, number]
ScalCoeffs = Union[float, complex, ndarray]  # admissable coefficents for VectorArray.scal/axpy

Coefficient = Union[RealOrComplex, 'ParameterFunctional']  # admissable linear coefficents for Operators
