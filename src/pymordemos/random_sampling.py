# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App

from pymor.tools.random import get_rng

app = App(help_on_error=True)


@app.default
def main(distribution: Literal['uniform', 'normal', 'weibull', 'gamma', 'exponential'],
         num_samples: int = 100):
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg

    p = thermal_block_problem((2, 1))
    fom, _ = discretize_stationary_cg(p, diameter=0.1)
    parameter_space = fom.parameters.space((0.1, 1))

    rng = get_rng()

    distributions = None
    if distribution == 'normal':
        distributions = {'diffusion': (lambda: rng.multivariate_normal(np.ones(2), np.diag([0.1, 0.25])).flatten())}
    elif distribution == 'weibull':
        distributions = {'diffusion': (lambda: rng.weibull(5, size=2))}
    elif distribution == 'gamma':
        distributions = {'diffusion': (lambda: rng.gamma(3, size=2))}
    elif distribution == 'exponential':
        distributions = {'diffusion': (lambda: rng.exponential(size=2))}

    parameters = parameter_space.sample_randomly(num_samples, distributions=distributions)

    parameters_array = np.array([mu.to_numpy() for mu in parameters])
    assert np.all((0.1 <= parameters_array) & (parameters_array <= 1.))
    plt.scatter(parameters_array[:, 0], parameters_array[:, 1])
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    app()
