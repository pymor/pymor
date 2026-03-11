# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.base import BasicObject


class BaseRegressor(BasicObject):
    def _get_extra_params(self):
        return dict()

    def get_params(self, deep=True):
        """Returns a dict of the init-parameters of the estimator, together with their values.

        The argument `deep=True` is required to match the scikit-learn interface.

        Parameters
        ----------
        deep
            If `True`, the parameters for this estimator and for the nested object
            (kernel, neural network, etc.) will be returned.

        Returns
        -------
        A dictionary of parameters and respective values of the estimator.
        """
        params = {name: getattr(self, name) for name in self._params}
        if deep and self._nested_object is not None:
            nested = getattr(self, self._nested_object)
            if hasattr(nested, 'get_params'):
                nested_object_params = nested.get_params(deep=True)
                for name, value in nested_object_params.items():
                    params[f'{self._nested_object}__{name}'] = value
        return params | self._get_extra_params()

    def _set_extra_param(self, key, param):
        raise NotImplementedError('`_set_extra_param` not available for this regressor')

    def set_params(self, **params):
        """Set the parameters of the estimator and the nested object.

        Supports nested parameter setting for the nested object using
        the value of ``self._nested_object`` as prefix.

        Parameters
        ----------
        params
            Estimator parameters to set.

        Returns
        -------
        An instance of the estimator with the new parameters.
        """
        if self._nested_object is not None:
            nested_object_params = {}
            prefix = self._nested_object + '__'

        for key, value in params.items():
            if self._nested_object is not None and key.startswith(prefix):
                nested_object_params[key.removeprefix(prefix)] = value
            elif key in self._params:
                setattr(self, key, value)
            else:
                self._set_extra_param(key, value)

        if self._nested_object is not None and nested_object_params:
            nested = getattr(self, self._nested_object)
            if not hasattr(nested, 'set_params'):
                raise ValueError('Nested object does not support parameter setting')
            nested.set_params(**nested_object_params)

        return self

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination on the data.

        Parameters
        ----------
        X
            Test samples for which to check the score.
        y
            Ground truth target values associated to the test samples.
        sample_weight
            Vector for weighting the different test samples.

        Returns
        -------
        The (weighted) coefficient of determination (:math:`R^2`-score) on the test samples.
        """
        y_pred = self.predict(X)

        if sample_weight is None:
            y_mean = np.mean(y)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
        else:
            sample_weight = np.asarray(sample_weight)
            y_mean = np.average(y, weights=sample_weight)
            ss_res = np.sum(sample_weight * (y - y_pred) ** 2)
            ss_tot = np.sum(sample_weight * (y - y_mean) ** 2)

        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
