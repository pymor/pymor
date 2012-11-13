
import abc
from pymor import core


class Interface(core.BaseInterface):

    @abc.abstractmethod
    def evaluate(self, x):
        pass

    def name(self):
        return 'common.function'

if __name__ == 'main':
    i = Interface()