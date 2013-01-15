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
    
    def __init__(self, function):
        self.decorated_function = function
        
    def __call__(self, im_self, *args, **kwargs):
        '''Via the magic that is partial functions returned from __get__, im_self is the instance object of the class
        we're decorating a method of and [kw]args are the actual parameters to the decorated method'''
        cache = im_self.cache_region
        keygen = im_self.keygen_generator(im_self.namespace, self.decorated_function)
        key = keygen(*args, **kwargs)
        def creator_function():
            self.logger.debug('creating new cache entry for {}.{}'
                              .format(im_self.__class__.__name__, self.decorated_function.__name__))
            return self.decorated_function(im_self, *args, **kwargs)
        return cache.get_or_create(key, creator_function, im_self.expiration_time)

    def __get__(self, instance, instancetype):
        """Implement the descriptor protocol to make decorating instance method possible.
        Return a partial function where the first argument is the instance of the decorated instance object.
        """ 
        return partial(self.__call__, instance)
        
        
class Cachable(object):
       
    def __init__(self, config=DEFAULT_MEMORY_CONFIG):
        self.cache_region = dc.make_region(function_key_generator = self.keygen_generator)
        self.cache_region.configure_from_config(config, '')
        self.namespace = '{}_{}'.format(self.__class__.__name__, hash(self))
        self.expiration_time = None
    
    def keygen_generator(self, namespace, function):
        '''I am the default generator function for (potentially) function specific keygens.
        I construct a key from the function name and given namespace 
        plus string representations of all positional and keyword args.
        '''
        fname = function.__name__
        namespace = str(namespace) 
        def keygen(*arg, **kwargs):
            return (namespace + "_" + fname + "_".join(str(s) for s in arg) 
                        + '__'.join(str(x) for x in kwargs.iteritems()))
        return keygen