# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
from hypothesis import assume, settings, HealthCheck
from hypothesis.strategies import sampled_from

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.rand_la import rrf 
from pymor.algorithms.basic import contains_zero_vector
from pymor.core.logger import log_levels
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import runmodule
from pymortests.strategies import given_vector_arrays

methods = [rrf]

def test_random_generalized_svd_without_products_vectorArray(vector_array, method):
    pass 

def test_random_generalized_svd_without_products_Operator(operator, method):
    pass 

def test_random_generalized_svd_with_products_Operator(operator, method):
    pass 

def test_random_generalized_svd_with_products_vectorArray(vector_array, method): 
    pass 

def test_random_generalized_svd_source_and_range_quadratic(vectorarray, method): 
    pass 

def test_p_not_to_large(vectorarray, method):
    pass 

def test_not_many_modes(method):
    pass 

def test_random_ghep_single_pass(operator, method):
    pass 

def test_random_ghep_NO_single_pass(operator, method): 
    pass




if __name__ == "__main__":
    runmodule(filename=__file__)
