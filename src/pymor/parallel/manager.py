# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import BasicObject


class RemoteObjectManager(BasicObject):
    """A simple context manager to keep track of |RemoteObjects|.

    When leaving this context, all |RemoteObjects| that have been
    :meth:`managed <manage>` by this object will be
    :meth:`removed <pymor.parallel.interface.RemoteObject.remove>`.
    """

    def __init__(self):
        self.remote_objects = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_objects()

    def __del__(self):
        self.remove_objects()

    def remove_objects(self):
        """Call :meth:`~pymor.parallel.interface.RemoteObject.remove` for all managed objects."""
        for obj in self.remote_objects:
            obj.remove()
        del self.remote_objects[:]

    def manage(self, remote_object):
        """Add a |RemoteObject| to the list of managed objects.

        Parameters
        ----------
        remote_object
            The object to add to the list.

        Returns
        -------
        `remote_object`
        """
        self.remote_objects.append(remote_object)
        return remote_object
