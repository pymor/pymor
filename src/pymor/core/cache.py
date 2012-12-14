import logging
import functools

from .timing import Timer

cache_miss_nesting = 0
cache_dt_nested = []

def cached(user_function):
    '''Our cache decorator. The current implementation is copied from
    functools32.lru_cache without the 'lru' functionality.
    '''

    hits, misses = [0], [0]
    kwd_mark = (object(),)          # separates positional and keyword args

    cache = dict()              # simple cache without ordering or size limit

    @functools.wraps(user_function)
    def wrapper(*args, **kwds):
        global cache_miss_nesting
        global cache_dt_nested
        key = args
        if kwds:
            key += kwd_mark + tuple(sorted(kwds.items()))
        try:
            result = cache[key]
            hits[0] += 1
            return result
        except KeyError:
            pass
        if len(cache_dt_nested) <= cache_miss_nesting:
            cache_dt_nested.append(0)
        msg = '|  ' * cache_miss_nesting + '/ CACHE MISS calling {}({},{})'.format(user_function.__name__, str(args), str(kwds))
        logging.debug(msg)

        cache_miss_nesting += 1
        timer = Timer('')
        timer.start()
        result = user_function(*args, **kwds)
        timer.stop()
        cache_miss_nesting -= 1

        if len(cache_dt_nested) > cache_miss_nesting + 1:
            dt_nested = cache_dt_nested[cache_miss_nesting + 1]
            cache_dt_nested[cache_miss_nesting + 1] = 0
        else:
            dt_nested = 0
        cache_dt_nested[cache_miss_nesting] += timer.dt

        msg = '|  ' * cache_miss_nesting + '\ call took {}s (own code:{}s, nested cache misses:{}s)'.format(timer.dt,
                timer.dt - dt_nested, dt_nested)
        logging.debug(msg)
        if cache_miss_nesting == 0:
            logging.debug('')
        cache[key] = result
        misses[0] += 1
        return result

    def cache_info():
        '''Report cache statistics'''
        return _CacheInfo(hits[0], misses[0], maxsize, len(cache))

    def cache_clear():
        '''Clear the cache and cache statistics'''
        cache.clear()
        hits[0] = misses[0] = 0

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper
