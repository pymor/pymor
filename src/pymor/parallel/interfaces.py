# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import BasicInterface, abstractmethod


class WorkerPoolInterface(BasicInterface):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def push(self, obj):
        pass

    @abstractmethod
    def scatter_array(self, U, copy=True):
        pass

    @abstractmethod
    def scatter_list(self, l):
        pass

    @abstractmethod
    def apply(self, function, *args, **kwargs):
        pass

    @abstractmethod
    def apply_only(self, function, worker, *args, **kwargs):
        pass

    @abstractmethod
    def map(self, function, *args, **kwargs):
        """Parallel version of the builtin map function.

        Each positional argument (after `function`) must be a sequence
        of same length n. `map` calls `function` in parallel on each of these n
        positional argument combinations, always passing `kwargs` as keyword
        arguments.  Keyword arguments which are |RemoteObjects| are automatically
        mapped to the respective object on the worker. Moreover, keyword arguments
        which are |immutable| objects that have already been pushed to the workers
        will not be transmitted again. (|Immutable| objects which have not
        been pushed before will be transmitted and the remote copy will be
        destroyed after function execution.)

        Parameters
        ----------
        function
            The function to execute on each worker.
        args
            The sequences of positional arguments for `function`.
        kwargs
            The keyword arguments for `function`.

        Returns
        -------
        List of return values of the function executions, ordered by
        the sequence of positional arguments.  If `function` returns
        a (tuple) of k values, k lists of return values are returned.
        """
        pass


class RemoteObjectInterface(object):

    removed = False

    @abstractmethod
    def _remove(self):
        pass

    def remove(self):
        if self.removed:
            return
        self._remove()
        self.removed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def __del__(self):
        self.remove()
