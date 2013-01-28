from __future__ import absolute_import, division, print_function, unicode_literals

class Named(object):

    __name = None
    @property
    def name(self):
        if self.__name is None:
            import uuid
            self.__name = str(uuid.uuid4())
        return self.__name

    @name.setter
    def name(self, n):
        if self.__name is not None:
            raise AttributeError('Name has already been set and cannot be changed')
        else:
            self.__name = n
