from __future__ import absolute_import, division, print_function

from itertools import izip

import numpy as np

import pymor.core as core
from pymor.core.interfaces import abstractmethod


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
        self.logger.info('Initial projection ...')
        self.data = self.initial_data()
        self.reduce(self.data)
        self.errors = []; self.max_err = 0; self.max_er_mu = 0;
        self.extensions = 0
        while not self.finished_after_extend():
            self.errors = [self.estimate(mu) for mu in samples]
            self.max_err, self.max_err_mu = max(((err, mu) for err, mu in izip(self.errors, samples)), key=lambda t:t[0])
            self.logger.info('Errors after {} extensions (max = {}):\n  {}\n'.format(self.extensions, self.max_err, self.errors))
            if self.finished_after_estimate():
                break
            self.logger.info('Extending with snapshot for mu = {}'.format(self.max_err_mu))
            self.data = self.extend(self.max_err_mu)
            self.reduce(self.data)
            self.extensions += 1
        return self.data


class GreedyRB(Greedy):

    def __init__(self, discretization, reductor):
        self.discretization = discretization
        self.reductor = reductor

    def reduce(self, data):
        self.rb_discretization, self.reconstructor = self.reductor.reduce(data)

    def initial_data(self):
        return np.zeros((0,self.discretization.solution_dim))

    def estimate(self, mu):
        U = self.discretization.solve(mu)
        URB = self.reconstructor.reconstruct(self.rb_discretization.solve(mu))
        return np.sqrt(np.sum((U-URB)**2)) / np.sqrt(np.sum(U**2))

    def extend(self, mu):
        U = self.discretization.solve(mu)
        new_data = np.empty((self.data.shape[0] + 1, self.data.shape[1]))
        new_data[:-1, :] = self.data
        new_data[-1, :] = U
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
        super(GreedyRB, self).run(samples)
