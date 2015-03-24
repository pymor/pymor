# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import BasicInterface, abstractmethod


class WorkerPoolInterface(BasicInterface):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def distribute(self, *args):
        pass

    @abstractmethod
    def apply(self, function, *args, **kwargs):
        pass

    @abstractmethod
    def apply_only(self, function, worker, *args, **kwargs):
        pass

    @abstractmethod
    def map(self, function, zip_return_values=True, *args, **kwargs):
        pass
