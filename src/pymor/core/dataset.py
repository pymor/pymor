from collections.abc import Iterable
from pathlib import Path

import numpy as np

from pymor.core.base import BasicObject
from pymor.core.cache import build_cache_key
from pymor.core.pickle import dump, load
from pymor.models.generic import GenericModel
from pymor.parameters.base import Parameters
from pymor.vectorarrays.interface import VectorArray


class DataSetReader(BasicObject):

    def __init__(self, path: str, id: str, parameters: Parameters, quantities: Iterable[str]):
        path = Path(path)
        assert isinstance(path, Path)
        assert path.exists(), f'Path {path} does not exist!'
        assert path.is_dir(), f'Path {path} is not a directory!'
        assert isinstance(id, str)
        id = id.strip()
        assert len(id) > 0
        assert isinstance(parameters, Parameters)
        assert isinstance(quantities, Iterable)
        assert all(isinstance(q, str) for q in quantities)
        quantities = set(quantities)
        self.__auto_init(locals())

    def _entry_prefix(self, mu):
        return f'{self.id}_{build_cache_key(mu)}'

    def _load(self, filename):
        with open(self.path / filename, 'rb') as f:
            return load(f)

    def get(self, mu, quantities=None):
        mu = self.parameters.parse(mu)
        if mu not in self:
            raise KeyError(f'Mu {mu} not found in dataset (id={id}).')
        base_filename = self._entry_prefix(mu)
        quantities = quantities or self.quantities
        self.logger.debug(f"Getting '{quantities}' for {mu} ...")
        if isinstance(quantities, str):
            assert quantities in self.quantities
            return self._load(f'{base_filename}_{quantities}.dat')
        else:
            assert isinstance(quantities, (tuple, list, set))
            assert all(isinstance(q, str) for q in quantities)
            assert all(q in self.quantities for q in quantities)
            data = {}
            for q in quantities:
                data[q] = self._load(f'{base_filename}_{q}.dat')
            return data

    def __contains__(self, mu):
        return (self.path / f'{self._entry_prefix(mu)}_mu.dat').exists()

    def keys(self):
        for p in self.path.glob(f'{self.id}*_mu.dat'):
            with open(p, 'rb') as f:
                yield load(f)

    def items(self, quantities=None):
        quantities = quantities or self.quantities
        if isinstance(quantities, str):
            assert quantities in self.quantities
        else:
            assert isinstance(quantities, (tuple, list, set))
            assert all(isinstance(q, str) for q in quantities)
            assert all(q in self.quantities for q in quantities)
        for mu in self.keys():
            yield mu, self.get(mu, quantities=quantities)

    def to_model(self, name=None):
        # we don't have the shapes or vector spaces, so we need to load an entry
        # as adding ensures conforming shapes, we can just pick the first entry
        _, data = next(iter(self.items()))

        def parse_shape(v):
            if isinstance(v, VectorArray):
                return v.space
            elif isinstance(v, np.ndarray):
                return v.shape[0]
            else:
                raise TypeError(f'Unsupported type {type(v)} for quantity value {v}. Expected VectorArray or np.ndarray!')
            
        def make_extractor(key):
            key = str(key)  # need to make a copy of the changing loop variable here
            return lambda mu: self.get(mu, key)

        return GenericModel(
            parameters=self.parameters,
            computers={
                q: (
                    parse_shape(v),
                    make_extractor(q),
                ) for q, v in data.items()
            },
            name=name or f'{self.id}FromDataSet',
        )

class DataSet(DataSetReader):

    def __init__(self, path: str, id: str, parameters: Parameters, quantities: Iterable[str]):
        assert isinstance(path, str)
        path = Path(path)
        assert isinstance(path, Path)
        path.mkdir(parents=True, exist_ok=True)
        super().__init__(path, id, parameters, quantities)
        self._seen_quantities = {q: None for q in self.quantities}
        self.__auto_init(locals())

    def _dump(self, v, filename):
        with open(self.path / filename, 'wb') as f:
            dump(v, f)

    def add(self, mu, data):
        mu = self.parameters.parse(mu)
        assert isinstance(data, dict)
        if mu in self:
            raise ValueError(f'An entry for {mu} already exists in this dataset!')
        base_filename = self._entry_prefix(mu)
        quantities_to_write = {}
        for q in self.quantities:
            assert q in data, f'Missing quantity {q} in data'
            v = data[q]
            if isinstance(v, VectorArray):
                if not self._seen_quantities[q]:
                    self._seen_quantities[q] = v.space
                assert v.space == self._seen_quantities[q]
            elif isinstance(v, np.ndarray):
                if not self._seen_quantities[q]:
                    self._seen_quantities[q] = v.shape
                assert v.shape == self._seen_quantities[q]
            else:
                raise TypeError(f'Unsupported type {type(v)} for quantity {q}. Expected VectorArray or np.ndarray!')
            quantities_to_write[q] = v
        self.logger.debug(f"Adding '{quantities_to_write.keys()}' for {mu} ...")
        for q, v in quantities_to_write.items():
            self._dump(v, f'{base_filename}_{q}.dat')
        # we write mu last, as it's the indicator for an existing entry
        self._dump(mu, f'{base_filename}_mu.dat')
