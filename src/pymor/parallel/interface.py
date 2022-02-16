# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import BasicObject, abstractmethod


class WorkerPool(BasicObject):
    """Interface for parallel worker pools.

    |WorkerPools| allow to easily parallelize algorithms which involve
    no or little communication between the workers at runtime. The interface
    methods give the user simple means to distribute data to
    workers (:meth:`~WorkerPool.push`, :meth:`~WorkerPool.scatter_array`,
    :meth:`~WorkerPool.scatter_list`) and execute functions on
    the distributed data in parallel (:meth:`~WorkerPool.apply`),
    collecting the return values from each function call. A
    single worker can be instructed to execute a function using the
    :meth:`WorkerPool.apply_only` method. Finally, a parallelized
    :meth:`~WorkerPool.map` function is available, which
    automatically scatters the data among the workers.

    All operations are performed synchronously.
    """

    @abstractmethod
    def __len__(self):
        """The number of workers in the pool."""
        pass

    @abstractmethod
    def push(self, obj):
        """Push a copy of `obj` to  all workers of the pool.

        A |RemoteObject| is returned as a handle to the pushed object.
        This object can be used as a keyword argument to :meth:`~WorkerPool.apply`,
        :meth:`~WorkerPool.apply_only`, :meth:`~WorkerPool.map`
        and will then be transparently mapped to the respective copy
        of the pushed object on the worker.

        |Immutable| objects will be pushed only once. If the same |immutable| object
        is pushed a second time, the returned |RemoteObject| will refer to the
        already transferred copy. It is therefore safe to use `push` to ensure
        that a given |immutable| object is available on the worker. No unnecessary
        copies will be created.

        Parameters
        ----------
        obj
            The object to push to all workers.

        Returns
        -------
        A |RemoteObject| referring to the pushed data.
        """
        pass

    @abstractmethod
    def scatter_array(self, U, copy=True):
        """Distribute |VectorArray| evenly among the workers.

        On each worker a |VectorArray| is created holding an (up to rounding) equal
        amount of vectors of `U`. The returned |RemoteObject| therefore refers
        to different data on each of the workers.

        Parameters
        ----------
        U
            The |VectorArray| to distribute.
        copy
            If `False`, `U` will be emptied during distribution of the vectors.

        Returns
        -------
        A |RemoteObject| referring to the scattered data.
        """
        pass

    @abstractmethod
    def scatter_list(self, l):
        """Distribute list of objects evenly among the workers.

        On each worker a `list` is created holding an (up to rounding) equal
        amount of objects of `l`. The returned |RemoteObject| therefore refers
        to different data on each of the workers.

        Parameters
        ----------
        l
            The list (sequence) of objects to distribute.

        Returns
        -------
        A |RemoteObject| referring to the scattered data.
        """
        pass

    @abstractmethod
    def apply(self, function, *args, **kwargs):
        """Apply function in parallel on each worker.

        This calls `function` on each worker in parallel, passing `args` as
        positional and `kwargs` as keyword arguments. Keyword arguments
        which are |RemoteObjects| are automatically mapped to the
        respective object on the worker. Moreover, keyword arguments which
        are |immutable| objects that have already been pushed to the workers
        will not be transmitted again. (|Immutable| objects which have not
        been pushed before will be transmitted and the remote copy will be
        destroyed after function execution.)

        Parameters
        ----------
        function
            The function to execute on each worker.
        args
            The positional arguments for `function`.
        kwargs
            The keyword arguments for `function`.

        Returns
        -------
        List of return values of the function executions, ordered by
        worker number (from `0` to `len(pool) - 1`).
        """
        pass

    @abstractmethod
    def apply_only(self, function, worker, *args, **kwargs):
        """Apply function on a single worker.

        This calls `function` on on the worker with number `worker`, passing
        `args` as positional and `kwargs` as keyword arguments. Keyword arguments
        which are |RemoteObjects| are automatically mapped to the
        respective object on the worker. Moreover, keyword arguments which
        are |immutable| objects that have already been pushed to the workers
        will not be transmitted again. (|Immutable| objects which have not
        been pushed before will be transmitted and the remote copy will be
        destroyed after function execution.)

        Parameters
        ----------
        function
            The function to execute.
        worker
            The worker on which to execute the function. (Number between
            `0` and `len(pool) - 1`.)
        args
            The positional arguments for `function`.
        kwargs
            The keyword arguments for `function`.

        Returns
        -------
        Return value of the function execution.
        """
        pass

    @abstractmethod
    def map(self, function, *args, **kwargs):
        """Parallel version of the builtin :func:`map` function.

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
        the sequence of positional arguments.
        """
        pass


class RemoteObject:
    """Handle to remote data on the workers of a |WorkerPool|.

    See documentation of :class:`WorkerPool` for usage
    of these handles in conjunction with :meth:`~WorkerPool.apply`,
    :meth:`~WorkerPool.scatter_array`,
    :meth:`~WorkerPool.scatter_list`.

    Remote objects can be used as a context manager: when leaving the
    context, the remote object's :meth:`~RemoteObject.remove`
    method is called to ensure proper cleanup of remote resources.

    Attributes
    ----------
    removed
        `True`, after :meth:`remove` has been called.
    """

    removed = False

    @abstractmethod
    def _remove(self):
        """Actual implementation of 'remove'."""
        pass

    def remove(self):
        """Remove the remote object from the workers.

        Remove the object this handle refers to from all workers.
        Note that the object will only be destroyed if no other
        object on the worker holds a reference to that object.
        Moreover, |immutable| objects will only be destroyed if
        :meth:`remove` has been called on *all* |RemoteObjects|
        which refer to the object (see :meth:`~WorkerPool.push`).
        """
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
