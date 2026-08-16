# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.interface import Model
from pymor.reductors.basic import AdaptiveReductor


class ModelHierarchy(Model):
    r"""Adaptive hierarchy of models of increasing fidelity.

    The hierarchy consists of a sequence of reduced models of increasing fidelity
    together with a high-fidelity reference model (typically the full-order model)
    that serves as the final fallback. For a new parameter, the cheapest model is
    evaluated first and the accuracy of its result is verified by means of an a
    posteriori error estimator. If the estimated error is smaller than the prescribed
    tolerance `tol`, the corresponding (reconstructed) solution is returned. Otherwise,
    the hierarchy falls back to the next more accurate model, until either a model is
    accurate enough or the reference model is reached.

    The hierarchy always tries the faster models first, until a solution is obtained
    that fulfills the accuracy requirement. Whenever the hierarchy has to fall back to
    a more accurate model, the cheaper models below it are adapted using the more
    accurate solution as training data (e.g. the reduced basis is extended or a
    data-driven surrogate is retrained). The hierarchy therefore starts with empty
    reduced models and improves them on demand. Models are only adapted when an actual
    solution or output is requested; queries for error estimates alone leave the
    hierarchy unchanged.

    This hierarchy is not restricted to a fixed set of models: any number of adaptive
    reductors can be combined, e.g. only a reduced basis reductor together with a
    full-order model, or a data-driven surrogate on top of a reduced basis model. The
    implementation is based on the strategies described in :cite:`HKOSW23`.

    Parameters
    ----------
    reductors
        Sequence of adaptive reductors ordered from the highest-fidelity reduced model
        down to the cheapest one (i.e. right below the reference model first). Each
        reductor has to provide `empty`, `reduce`, `reconstruct` and `adapt` methods
        (see :class:`~pymor.reductors.basic.ProjectionBasedReductor` and
        :class:`~pymor.reductors.data_driven.AdaptiveDDReductor`).
    tol
        Tolerance against which the estimated errors are compared to decide which
        model's solution to return.
    models
        Optional sequence of already constructed models, ordered from the reference
        model down to the cheapest reduced model (`len(reductors) + 1` entries). If not
        provided, the models are constructed from `reductors` and `fom`.
    fom
        The high-fidelity reference model used as the final fallback. Required if
        `models` is not given.
    time_reduction
        Callable mapping a (possibly time-dependent) error estimate to a single scalar
        that is compared against `tol`. For instationary problems the error estimate is a
        trajectory over time; the default `numpy.max` uses the maximum in time (i.e. the
        :math:`\ell^\infty`-in-time norm). Other choices (e.g. the value at the final time
        or an :math:`\ell^2`-in-time norm) can be passed here.
    """

    def __init__(self, reductors, tol, models=None, fom=None, time_reduction=np.max):
        assert len(reductors) >= 1
        assert tol > 0
        assert callable(time_reduction)
        assert models is not None or fom is not None
        # make sure that all reductors fulfill the `AdaptiveReductor`-interface
        assert all(isinstance(red, AdaptiveReductor) for red in reductors)

        # the reductors list is reversed internally to simplify the iteration over the reductors
        reductors = list(reversed(reductors))

        # set up the models list either using the provided models or the given reductors
        if models is not None:
            assert len(models) == len(reductors) + 1
            models = list(reversed(models))
        else:
            models = [red.reduce() for red in reductors] + [fom]

        # use the last model as reference model
        reference_model = models[-1]

        super().__init__(dim_input=reference_model.dim_input, products=reference_model.products,
                         visualizer=reference_model.visualizer)
        self.solution_space = reference_model.solution_space
        self.dim_output = reference_model.dim_output
        self.__auto_init(locals())

    def _select_model(self, mu, quantities):
        base_quantities = quantities & {'solution', 'output'}

        errors_to_compute = set()
        if quantities & {'solution', 'solution_error_estimate'}:
            errors_to_compute.add('solution_error_estimate')
        if quantities & {'output', 'output_error_estimate'}:
            errors_to_compute.add('output_error_estimate')

        def below_tol(estimate):
            return estimate is None or self.time_reduction(estimate) <= self.tol

        # iterate over the models to determine sufficiently accurate solution and/or output
        for i_m, m in enumerate(self.models):
            # check if the current model is the reference model
            is_reference = i_m == len(self.models) - 1

            # skip the current model if the corresponding reductor is empty
            if not is_reference and self.reductors[i_m].empty:
                continue

            # determine the requested quantities and call the `compute`-method of the current model
            requested = base_quantities if is_reference else base_quantities | errors_to_compute
            result = m.compute(**dict.fromkeys(requested, True), mu=mu)

            # except the result by leaving the loop if the reference model was reached
            # or the estimated errors in solution and output are below the tolerance
            if is_reference or (below_tol(result.get('solution_error_estimate'))
                                and below_tol(result.get('output_error_estimate'))):
                break

        i_m_sufficient = i_m

        return result, i_m_sufficient

    def _fill_zero_error_estimates(self, data, quantities):
        error_estimates = {'solution_error_estimate', 'output_error_estimate'}
        # the reference model is exact: report a zero error, shaped like the corresponding
        # estimate, for any requested estimate that no model produced
        for quantity in quantities & error_estimates:
            if quantity in data:
                continue
            if quantity == 'solution_error_estimate' and 'solution' in data:
                data[quantity] = np.zeros(len(data['solution']))
            elif quantity == 'output_error_estimate' and 'output' in data:
                data[quantity] = np.zeros_like(data['output'])
            else:
                data[quantity] = np.zeros(1)

    def _reconstruct(self, data, i_m_sufficient, quantities, result):
        data['used_model'] = len(self.models) - 1 - i_m_sufficient
        # if the solution is requested, reconstruct it such that it lives
        # in the solution space of the reference model
        if 'solution' in quantities:
            solution = result['solution']
            # iteratively reconstruct through all reductors in the hierarchy starting from the one
            # whose model was the first to produce a sufficiently accurate solution
            for red in self.reductors[i_m_sufficient:]:
                solution = red.reconstruct(solution)  # could be a nop
            data['solution'] = solution

    def _adapt_lower_fidelity_models(self, mu, i_m_sufficient, result):
        m_new, adapt_data = self.reductors[i_m_sufficient-1].adapt(mu, fom_solution=result.get('solution'),
                                                                   fom_output=result.get('output'))
        if self.models[i_m_sufficient-1] == m_new:
            return
        self.models[i_m_sufficient-1] = m_new

        # iteratively adapt the lower fidelity models using the higher fidelity data
        # from the previous model in the hierarchy
        for i in range(i_m_sufficient-2, -1, -1):
            # call the `adapt`-method of the respective `AdaptiveReductor`
            m_new, adapt_data = self.reductors[i].adapt(mu, new_fom=m_new,
                                                        fom_solution=adapt_data.get('solution'),
                                                        fom_output=adapt_data.get('output'))
            # end the loop if the model did not change
            if self.models[i] == m_new:
                return
            # update the model in the models list
            self.models[i] = m_new

    def _compute(self, quantities, data, mu):
        result, i_m_sufficient = self._select_model(mu, quantities)
        data.update(result)

        self._fill_zero_error_estimates(data, quantities)

        self._reconstruct(data, i_m_sufficient, quantities, result)

        base_quantities = quantities & {'solution', 'output'}
        if not base_quantities or i_m_sufficient == 0:
            return

        self._adapt_lower_fidelity_models(mu, i_m_sufficient, result)

    def retrain(self):
        """Manually retrain all reductors that support it and refresh the active models.

        A manual alternative to the automatic retraining controlled by a reductor's
        retraining interval: every reductor providing a `retrain` method (e.g.
        :class:`~pymor.reductors.data_driven.AdaptiveDDReductor`) is retrained on the
        training data collected so far, and the corresponding model in the hierarchy is
        replaced by the retrained one.
        """
        for i, reductor in enumerate(self.reductors):
            if hasattr(reductor, 'retrain'):
                self.models[i] = reductor.retrain()
