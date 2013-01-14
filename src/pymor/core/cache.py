from functools import wraps, partial
from dogpile import cache as dc
from os.path import join
from tempfile import gettempdir
import mock

from pymor.core.interfaces import abstractmethod, BasicInterface

DEFAULT_MEMORY_CONFIG = { "backend":'dogpile.cache.memory' }
DEFAULT_DISK_CONFIG = { "backend":'dogpile.cache.dbm',
        "arguments.filename": join(gettempdir(), 'pymor.cache.dbm')}


class cached(BasicInterface):
    
    def __init__(self, f):
        self.f = f
        self.cache = {}
        self.expiration_time = None

    def __call__(self, im_self, *args, **kwargs):
        cache = im_self._cache_region
        keygen = im_self._keygen(im_self._namespace, self.f)
        key = keygen(*args, **kwargs)
        self.logger.debug('self {} key {} args {} kwargs {}'.format(im_self, key, str(*args), str(**kwargs)))
        def creator():
            return self.f(im_self, *args, **kwargs)
        return cache.get_or_create(key, creator, self.expiration_time)

    def __get__(self, instance, instancetype):
        """Implement the descriptor protocol to make decorating instance method possible.
        Return a partial function where the first argument is the instance of the decorated class.
        """ 
        return partial(self.__call__, instance)
        
        
class Cachable(object):
       
    def __init__(self, config=DEFAULT_MEMORY_CONFIG):
        self._cache_region = dc.make_region(function_key_generator = self._keygen)
        self._cache_region.configure_from_config(config, '')
        self._namespace = self.__class__.__name__
    
    def _keygen(self, namespace, function):
        '''
        '''
        fname = function.__name__
        namespace = str(namespace) 
        def generate_key(*arg):
            return namespace + "_" + fname + "_".join(str(s) for s in arg)
        return generate_key