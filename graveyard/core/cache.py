# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


import pymor.core.dogpile_backends


class DogpileCacheRegion(CacheRegion):

    def get(self, key):
        value = self._cache_region.get(key)
        if value is pymor.core.dogpile_backends.NO_VALUE:
            return False, None
        else:
            return True, value

    def set(self, key, value):
        self._cache_region.set(key, value)


class DogpileMemoryCacheRegion(DogpileCacheRegion):

    def __init__(self):
        self._new_region()

    def _new_region(self):
        from dogpile import cache as dc
        self._cache_region = dc.make_region()
        self._cache_region.configure_from_config(pymor.core.dogpile_backends.DEFAULT_MEMORY_CONFIG, '')

    def clear(self):
        self._new_region()

    def set(self, key, value):
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        self._cache_region.set(key, value)


class DogpileDiskCacheRegion(DogpileCacheRegion):

    def __init__(self, filename=None, max_size=1024 ** 3):
        self.filename = filename
        self.max_size = max_size
        self._new_region()

    def _new_region(self):
        from dogpile import cache as dc
        self._cache_region = dc.make_region()
        config = dict(pymor.core.dogpile_backends.DEFAULT_DISK_CONFIG)
        if self.filename:
            config['arguments.filename'] = os.path.expanduser(self.filename)
        if self.max_size:
            config['arguments.max_size'] = self.max_size
        self._cache_region.configure_from_config(config, '')

    def clear(self):
        import glob
        filename = self._cache_region.backend.filename
        del self._cache_region
        files = glob.glob(filename + '*')
        map(os.unlink, files)
        self._new_region()
