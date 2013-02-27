from __future__ import absolute_import, division, print_function

class Named(object):

    __name = None
    @property
    def name(self):
        if self.__name is None:
            import uuid
            self.__name = '{}_{}'.format(self.__class__.__name__, uuid.uuid4())
        return self.__name

    @name.setter
    def name(self, n):
        self.__name = n
