from __future__ import absolute_import, division, print_function
import pkgutil

import pymor.core as core
from pymortests.base import runmodule


def exercise_logger(logger):
    for lvl in core.logger.LOGLEVEL_MAPPING.values():
        logger.setLevel(lvl)
        assert logger.isEnabledFor(lvl)
    for verb in ['warn', 'error', 'debug', 'info']:
        getattr(logger, verb)('{} -- logger {}'.format(verb, str(logger)))


def logmodule(module_name):
    logger = core.getLogger(module_name)
    exercise_logger(logger)


def logclass(cls):
    logger = cls.logger
    exercise_logger(logger)


def test_logger():
    import pymor
    fails = []
    for importer, pack_name, _ in pkgutil.walk_packages(pymor.__path__, pymor.__name__ + '.',
                                                        lambda n: fails.append(n)):
        yield logmodule, pack_name
        try:
            importer.find_module(pack_name).load_module(pack_name)
        except TypeError, e:
            fails.append(pack_name)
    import pprint
    if len(fails):
        core.getLogger(__name__).error('Failed imports: {}'.format(pprint.pformat(set(fails))))
    for cls in pymor.core.interfaces.BasicInterface.implementors(True):
        yield logclass, cls


if __name__ == "__main__":
    runmodule(name='pymortests.core.logger')
