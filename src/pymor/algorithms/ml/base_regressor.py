# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.ml.base_estimator import BaseEstimator


class BaseRegressor(BaseEstimator):
    """Base-class for scikit-learn-style regressors.

    This class provides a `score` function, computing the coefficient of
    determination.
    This class cannot be used directly but serves as a base class for
    other regressors.
    """

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

        y = np.asarray(y)
        y_pred = np.asarray(y_pred).reshape(y.shape)

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
