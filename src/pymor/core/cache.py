from functools import wraps


def cached(user_function):
    '''Our cache decorator. The current implementation is copied from
    functools32.lru_cache without the 'lru' functionality.
    '''

    hits, misses = [0], [0]
    kwd_mark = (object(),)          # separates positional and keyword args

    cache = dict()              # simple cache without ordering or size limit

    @wraps(user_function)
    def wrapper(*args, **kwds):
        key = args
        if kwds:
            key += kwd_mark + tuple(sorted(kwds.items()))
        try:
            result = cache[key]
            hits[0] += 1
            return result
        except KeyError:
            pass
        result = user_function(*args, **kwds)
        cache[key] = result
        misses[0] += 1
        return result

    def cache_info():
        """Report cache statistics"""
        return _CacheInfo(hits[0], misses[0], maxsize, len(cache))

    def cache_clear():
        """Clear the cache and cache statistics"""
        cache.clear()
        hits[0] = misses[0] = 0

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper
