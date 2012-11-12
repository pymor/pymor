
import abc
from pymor import core

class Interface(core.BaseInterface):

    def name(self):
        return 'problem.stationary.linear.elliptic'
