# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.basic import (OperatorBase, AssemblableOperatorBase, LincombOperatorBase, LincombOperator,
                                   NumpyGenericOperator, NumpyMatrixBasedOperator, NumpyMatrixOperator)
from pymor.operators.constructions import ConstantOperator, FixedParameterOperator, VectorOperator, VectorFunctional
from pymor.operators.interfaces import OperatorInterface, LincombOperatorInterface
