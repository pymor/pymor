# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np

import pymor.core as core
from pymor.core.interfaces import abstractmethod
from pymor.la import gram_schmidt, l2_norm


class Greedy(core.BasicInterface):

    @abstractmethod
    def initial_data(self):
        pass

    @abstractmethod
    def reduce(self, data):
        pass

    @abstractmethod
    def estimate(self, mu):
        pass

    @abstractmethod
    def extend(self, mu):
        pass

    def finished_after_estimate(self):
        return False

    def finished_after_extend(self):
        return False

    def run(self, samples):

        samples = list(samples)
        self.logger.info('Started greedy search on samples\n  {}'.format([str(mu) for mu in samples]))
        self.samples = samples
        self.additional_info = {}
        self.logger.info('Initial projection')
        self.data = self.initial_data()
        self.reduce(self.data)
        self.errors = []
        self.max_err = 0
        self.max_er_mu = 0
        self.extensions = 0
        while not self.finished_after_extend():
            self.logger.info('Estimating errors ...')
            self.errors = [self.estimate(mu) for mu in samples]
            self.max_err, self.max_err_mu = max(((err, mu)
                                                for err, mu in izip(self.errors, samples)), key=lambda t: t[0])
            self.logger.info('Maximal errors after {} extensions: {} (mu = {})\n'.format(self.extensions, self.max_err,
                                                                                         self.max_err_mu))
            if self.finished_after_estimate():
                break
            self.logger.info('Extending with snapshot for mu = {}'.format(self.max_err_mu))
            self.data = self.extend(self.max_err_mu)
            self.reduce(self.data)
            self.extensions += 1
        return self.data


class GreedyRB(Greedy):

    def __init__(self, discretization, reductor, use_estimator=True, error_norm=l2_norm,
                 extension_algorithm='gram_schmidt'):
        assert extension_algorithm in ('trivial', 'gram_schmidt')
        self.discretization = discretization
        self.reductor = reductor
        self.use_estimator = use_estimator
        self.error_norm = l2_norm
        self.extension_algorithm = extension_algorithm
        self.estimator_warning = False

    def reduce(self, data):
        self.rb_discretization, self.reconstructor = self.reductor.reduce(data)
        self.estimator_warning = False

    def initial_data(self):
        self.basis_enlarged = True
        return np.zeros((0, self.discretization.solution_dim))

    def estimate(self, mu):
        est = self.use_estimator
        if self.use_estimator and not hasattr(self.rb_discretization, 'estimate'):
            if not self.estimator_warning:
                self.logger.warn('Reduced discretization has no estimator, computing detailed solution')
                self.estimator_warning = True
            est = False

        if est:
            return self.rb_discretization.estimate(self.rb_discretization.solve(mu), mu)
        else:
            U = self.discretization.solve(mu)
            URB = self.reconstructor.reconstruct(self.rb_discretization.solve(mu))
            return self.error_norm(U - URB)

    def extend(self, mu):
        U = self.discretization.solve(mu)
        new_data = np.empty((self.data.shape[0] + 1, self.data.shape[1]))
        new_data[:-1, :] = self.data
        new_data[-1, :] = U
        if self.extension_algorithm == 'gram_schmidt':
            new_data = gram_schmidt(new_data, row_offset=self.data.shape[0])
        self.basis_enlarged = (new_data.size > self.data.size)
        if self.basis_enlarged:
            self.logger.info('Extended basis to size {}'.format(new_data.shape[0]))
        return new_data

    def finished_after_estimate(self):
        if self.target_err is not None:
            if self.max_err <= self.target_err:
                self.logger.info('Reached maximal error on snapshots of {} <= {}'.format(self.self.max_err,
                                                                                         self.target_err))
                return True
            else:
                return False
        else:
            return False

    def finished_after_extend(self):
        if not self.basis_enlarged:
            self.logger.info('Failed to enlarge basis. Stopping now.')
            return True
        if self.Nmax is not None:
            if self.data.shape[0] >= self.Nmax:
                self.logger.info('Reached maximal basis size of {} vectors'.format(self.Nmax))
                return True
            else:
                return False
        else:
            return False

    def run(self, samples, Nmax=None, err=None):
        assert Nmax is not None or err is not None
        self.Nmax = Nmax
        self.target_err = err
        return super(GreedyRB, self).run(samples)
