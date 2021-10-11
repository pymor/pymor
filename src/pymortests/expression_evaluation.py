import numpy as np

from pymor.analyticalproblems.functions import ExpressionFunction
from pymortests.base import runmodule


def test_two_sided_expression_evaluation():
    f = ExpressionFunction("(-2 < x[0] < 2) * 1.", 1)

    zeros = [-3, -2, 2, 3]      # values for which f should return array(0.)
    ones = [-1, 0, 1]           # values for which f should return array(1.)

    values = []
    for eval in zeros + ones:
        values.append(f([eval]).item())

    assert np.array_equal(values[:len(zeros)], [0. for i in zeros]), 'Two sided comparison failed for zero values!'
    assert np.array_equal(values[len(zeros):], [1. for i in ones]), 'Two sided comparison failed for one values!'


def test_keyword_not_expression_evaluation():
    f = ExpressionFunction("(not (0 < x[0])) * 1.", 1)

    zeros = [1]                 # values for which f should return array(0.)
    ones = [0, -1]              # values for which f should return array(1.)

    values = []
    for eval in zeros + ones:
        values.append(f([eval]).item())

    assert np.array_equal(values[:len(zeros)], [0. for i in zeros]), 'Keyword "not" evaluation failed for zero values!'
    assert np.array_equal(values[len(zeros):], [1. for i in ones]), 'Keyword "not" evaluation failed for zero values!'


if __name__ == "__main__":
    runmodule(filename=__file__)
