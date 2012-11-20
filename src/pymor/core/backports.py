'''This module contains pure python backports of library features from python >= 3 to 2.7
'''

class abstractclassmethod(classmethod):
    """
    A decorator indicating abstract classmethods.

    Similar to abstractmethod.

    Usage:

        class C(metaclass=ABCMeta):
            @abstractclassmethod
            def my_abstract_classmethod(cls, ...):
                ...

    'abstractclassmethod' is deprecated. Use 'classmethod' with
    'abstractmethod' instead.
    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


class abstractstaticmethod(staticmethod):
    """
    A decorator indicating abstract staticmethods.

    Similar to abstractmethod.

    Usage:

        class C(metaclass=ABCMeta):
            @abstractstaticmethod
            def my_abstract_staticmethod(...):
                ...

    'abstractstaticmethod' is deprecated. Use 'staticmethod' with
    'abstractmethod' instead.
    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractstaticmethod, self).__init__(callable)