# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import matplotlib.pyplot as plt
import numpy as np

from pymor.models.interface import Model
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.reductors.data_driven import DataDrivenReductor


class DDRBModelHierarchy(Model):
    def __init__(self, fom, rb_reductor, dd_reductor_parameters, tol):
        self.__auto_init(locals())
        self.__dict__.pop('dd_model', None)

        assert isinstance(rb_reductor, ProjectionBasedReductor)

        self.dd_reductors = []
        self.dd_models = []
        self._rb_model = self.rb_reductor.reduce()

        self.used_model = []
        self.estimated_errors = []

        super().__init__()

    def print_summary(self):
        print('Number of successful calls:')
        print(f"\tFOM: {self.used_model.count('FOM')}\t"
              f"(ratio: {self.used_model.count('FOM')/len(self.used_model)*100:.2f}%)")
        print(f"\tRB: {self.used_model.count('RB')}\t"
              f"(ratio: {self.used_model.count('RB')/len(self.used_model)*100:.2f}%)")
        print(f"\tML: {self.used_model.count('ML')}\t"
              f"(ratio: {self.used_model.count('ML')/len(self.used_model)*100:.2f}%)")

    def plot_summary(self):
        def get_indices(element, lst):
            return [i for i in range(len(lst)) if lst[i] == element]

        axs = plt.subplot()
        axs.set_xlabel('parameter index')
        axs.set_ylabel('error estimate')
        axs.plot(get_indices('ML', self.used_model),
                 np.array(self.estimated_errors)[get_indices('ML', self.used_model)],
                 '*', label='ML')
        axs.plot(get_indices('RB', self.used_model),
                 np.array(self.estimated_errors)[get_indices('RB', self.used_model)],
                 '.', label='RB')
        axs.legend()
        axs.semilogy()
        axs.set_title('Estimated errors')
        plt.show()

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities:
            if len(self.dd_models) == 0:
                dd_solution = self._rb_model.solution_space.zeros()
            else:
                dd_solution_np = np.vstack([dd_model.solve(mu).to_numpy() for dd_model in self.dd_models])
                dd_solution = self._rb_model.solution_space.make_array(dd_solution_np)
            dd_estimated_error = self._rb_model.error_estimator.estimate_error(dd_solution, mu, self._rb_model)
            self.logger.info(f'Estimated error of ML: {dd_estimated_error}')

            if dd_estimated_error <= self.tol:
                data['solution'] = self.rb_reductor.reconstruct(dd_solution)

                self.used_model.append('ML')
                self.estimated_errors.append(dd_estimated_error[0])
            else:
                rb_solution = self._rb_model.solve(mu)
                rb_estimated_error = self._rb_model.error_estimator.estimate_error(rb_solution, mu, self._rb_model)
                self.logger.info(f'Estimated error of RB: {rb_estimated_error}')
                if rb_estimated_error <= self.tol:
                    data['solution'] = self.rb_reductor.reconstruct(rb_solution)
                    # Extend training data of `dd_reductors`
                    sum_dims = 0
                    for i, red in enumerate(self.dd_reductors):
                        red.extend_training_data([mu], rb_solution.to_numpy()[sum_dims:sum_dims+red.dim_solution_space])
                        self.dd_models[i] = red.reduce()
                        sum_dims += red.dim_solution_space

                    self.used_model.append('RB')
                    self.estimated_errors.append(rb_estimated_error[0])
                else:
                    fom_solution = self.fom.solve(mu)
                    data['solution'] = fom_solution
                    # Extend reduced basis of `rb_reductor`
                    # TODO: For instationary problems:
                    # compress solution trajectory first before extending the basis!
                    old_rb_size = len(self.rb_reductor.bases['RB'])
                    self.rb_reductor.extend_basis(fom_solution)
                    self._rb_model = self.rb_reductor.reduce()
                    projected_fom_solution = self.rb_reductor.bases['RB'].inner(fom_solution)

                    sum_dims = 0
                    for i, red in enumerate(self.dd_reductors):
                        red.extend_training_data([mu], projected_fom_solution[sum_dims:sum_dims+red.dim_solution_space])
                        self.dd_models[i] = red.reduce()
                        sum_dims += red.dim_solution_space

                    # Add new dd_model to `self.dd_models`
                    # in order to account for new basis components
                    self.dd_reductors.append(DataDrivenReductor([mu], projected_fom_solution[old_rb_size:],
                                                                **self.dd_reductor_parameters))
                    self.dd_models.append(self.dd_reductors[-1].reduce())

                    self.used_model.append('FOM')
                    self.estimated_errors.append(0.)

            quantities.remove('solution')

        super()._compute(quantities, data, mu=mu)
