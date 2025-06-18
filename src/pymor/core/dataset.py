from collections.abc import Iterable
from pathlib import Path

import numpy.typing as npt

from pymor.core.base import BasicObject
from pymor.core.cache import build_cache_key
from pymor.core.pickle import dump, load
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

    def get(self, mu, quantities=None):
        mu = self.parameters.parse(mu)
        if mu not in self:
            raise KeyError(f'Mu {mu} not found in dataset (id={id}).')
        base_filename = self._entry_prefix(mu)
        quantities = quantities or self.quantities
        if isinstance(quantities, str):
            assert quantities in self.quantities
            with open(self.path / f'{base_filename}_{quantities}.dat', 'rb') as f:
                return load(f)
        else:
            assert isinstance(quantities, (tuple, list, set))
            assert all(isinstance(q, str) for q in quantities)
            assert all(q in self.quantities for q in quantities)
            data = {}
            for q in quantities:
                with open(self.path / f'{base_filename}_{q}.dat', 'rb') as f:
                    data[q] = load(f)
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


class DataSet(DataSetReader):

    def __init__(self, path: str, id: str, parameters: Parameters, quantities: Iterable[str]):
        assert isinstance(path, str)
        path = Path(path)
        assert isinstance(path, Path)
        path.mkdir(parents=True, exist_ok=True)
        super().__init__(path, id, parameters, quantities)
        self._seen_quantities = dict.fromkeys(self.quantities)
        self.__auto_init(locals())

    def add(self, mu, data):
        mu = self.parameters.parse(mu)
        assert isinstance(data, dict)
        if mu in self:
            raise ValueError(f'An entry for {mu} already exists in this dataset!')
        base_filename = self._entry_prefix(mu)
        for q in self.quantities:
            assert q in data, f'Missing quantity {q} in data'
            v = data[q]
            if isinstance(v, VectorArray):
                if not self._seen_quantities[q]:
                    self._seen_quantities[q] = v.space
                assert v.space == self._seen_quantities[q]
            elif isinstance(v, npt.ArrayLike):
                if not self._seen_quantities[q]:
                    self._seen_quantities[q] = v.shape
                assert v.shape == self._seen_quantities[q]
            else:
                raise TypeError(f'Unsupported type {type(v)} for quantity {q}. Expected VectorArray or npt.ArrayLike!')
            with open(self.path / f'{base_filename}_{q}.dat', 'wb') as f:
                dump(v, f)
        # we write mu last, as it's the indicator for an existing entry
        with open(self.path / f'{base_filename}_mu.dat', 'wb') as f:
            dump(mu, f)
