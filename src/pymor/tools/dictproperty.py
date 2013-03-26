from __future__ import absolute_import, division, print_function


def dict_property(d, n):
    '''A property which takes its value from self.d[n].'''
    @property
    def prop(self):
        return self.__dict__[d][n]

    @prop.setter
    def prop(self, v):
        self.__dict__[d][n] = v

    return prop
