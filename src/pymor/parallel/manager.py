# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import BasicInterface


class RemoteObjectManager(BasicInterface):

    def __init__(self):
        self.remote_objects = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_objects()

    def __del__(self):
        self.remove_objects()

    def remove_objects(self):
        for obj in self.remote_objects:
            obj.remove()
        del self.remote_objects[:]

    def manage(self, remote_object):
        self.remote_objects.append(remote_object)
        return remote_object
