# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os
import pymordemos
import runpy
import sys
import pytest
from tempfile import mkdtemp
import shutil

from pymortests.base import runmodule, check_results
from pymor.core.exceptions import QtMissing, GmshMissing, MeshioMissing
from pymor.discretizers.builtin.gui.qt import stop_gui_processes
from pymor.core.config import is_windows_platform, is_macos_platform
from pymor.tools.mpi import parallel


DISCRETIZATION_ARGS = (
    ('elliptic', [0, 0, 0, 0]),
    ('elliptic', [1, 2, 0, 3]),
    ('elliptic', ['--rect', 1, 2, 0, 3]),
    ('elliptic', [0, 0, 2, 1]),
    ('elliptic', ['--fv', 0, 0, 0, 0]),
    ('elliptic', ['--fv', 1, 2, 0, 3]),
    ('elliptic', ['--fv', '--rect', 1, 2, 0, 3]),
    ('elliptic', ['--fv', 0, 0, 2, 1]),
    ('elliptic2', [0, 20]),
    ('elliptic2', ['--fv', 0, 20]),
    ('elliptic2', [1, 20]),
    ('elliptic_oned', [1, 20]),
    ('elliptic_oned', ['--fv', 1, 20]),
    ('burgers', ['--num-flux=lax_friedrichs', '0.1']),
    ('burgers', ['--num-flux=engquist_osher', '0.1']),
    ('burgers', ['--num-flux=simplified_engquist_osher', '0.1']),
    ('parabolic', ['heat', 1]),
    ('parabolic', ['heat', '--rect', 1]),
    ('parabolic', ['heat', '--fv', 1]),
    ('parabolic', ['heat', '--rect', '--fv', 1]),
    ('parabolic', ['dar', 1]),
    ('parabolic', ['dar', '--rect', 1]),
)

if not parallel:
    DISCRETIZATION_ARGS += (('elliptic_unstructured', [6., 16, 1e-1]),)

THERMALBLOCK_ARGS = (
    ('thermalblock', ['--plot-solutions', '--plot-err', '--plot-error-sequence', 2, 2, 3, 5]),
    ('thermalblock', ['--fenics', 2, 2, 3, 5]),
    ('thermalblock', ['--greedy-without-estimator', 3, 1, 2, 5]),
    ('thermalblock', ['--alg=pod', 2, 2, 3, 5]),
    ('thermalblock', ['--alg=adaptive_greedy', 2, 2, 10, 30]),
    ('thermalblock', ['--alg=naive', '--reductor=traditional', 2, 2, 10, 5]),
)

THERMALBLOCK_ADAPTIVE_ARGS = (
    ('thermalblock_adaptive', [10]),
    ('thermalblock_adaptive', ['--no-visualize-refinement', 10]),
)

THERMALBLOCK_SIMPLE_ARGS = (
    ('thermalblock_simple', ['pymor', 'naive', 2, 10, 10]),
    ('thermalblock_simple', ['fenics', 'greedy', 2, 10, 10]),
    ('thermalblock_simple', ['ngsolve', 'pod', 2, 10, 10]),
    ('thermalblock_simple', ['pymor-text', 'adaptive_greedy', -1, 3, 3]),
)

THERMALBLOCK_GUI_ARGS = (
    ('thermalblock_gui', ['--testing', 2, 2, 3, 5]),
)

BURGERS_EI_ARGS = (
    ('burgers_ei', [1, 2, 2, 5, 2, 5, '--plot-ei-err']),
    ('burgers_ei', [1, 2, 2, 5, 2, 5, '--ei-alg=deim']),
)

PARABOLIC_MOR_ARGS = (
    ('parabolic_mor', ['pymor', 'greedy', 2, 3, 1]),
    ('parabolic_mor', ['pymor', 'pod', 2, 3, 1]),
    ('parabolic_mor', ['fenics', 'adaptive_greedy', 2, 3, 1]),
)

SYS_MOR_ARGS = (
    ('heat', []),
    ('delay', []),
    ('string_equation', [])
)

HAPOD_ARGS = (
    ('hapod', ['--snap=3', 1e-2, 10, 100]),
    ('hapod', ['--snap=3', '--threads=2', 1e-2, 10, 100]),
    ('hapod', ['--snap=3', '--procs=2', 1e-2, 10, 100]),
)

FENICS_NONLINEAR_ARGS = (
    ('fenics_nonlinear', [2, 10, 2]),
    ('fenics_nonlinear', [3, 5, 1]),
)

DEMO_ARGS = (
    DISCRETIZATION_ARGS
    + THERMALBLOCK_ARGS
    + THERMALBLOCK_ADAPTIVE_ARGS
    + THERMALBLOCK_SIMPLE_ARGS
    + THERMALBLOCK_GUI_ARGS
    + BURGERS_EI_ARGS
    + PARABOLIC_MOR_ARGS
    + SYS_MOR_ARGS
    + HAPOD_ARGS
    + FENICS_NONLINEAR_ARGS
)
DEMO_ARGS = [(f'pymordemos.{a}', b) for (a, b) in DEMO_ARGS]


def _run_module(module, args):
    sys.argv = [module] + [str(a) for a in args]
    return runpy.run_module(module, init_globals=None, run_name='__main__', alter_sys=True)


def _skip_if_no_solver(param):
    demo, args = param
    from pymor.core.config import config
    for solver in ['fenics', 'ngsolve']:
        needs_solver = len([f for f in args if solver in str(f)]) > 0 or demo.find(solver) >= 0
        has_solver = getattr(config, 'HAVE_' + solver.upper())
        if needs_solver and not has_solver:
            if not os.environ.get('DOCKER_PYMOR', False):
                pytest.skip('skipped test due to missing ' + solver)
    if demo == 'heat' and not os.environ.get('DOCKER_PYMOR', False) and not config.HAVE_SLYCOT:
        pytest.skip('skipped test due to missing slycot')


def _demo_ids(demo_args):
    def _key(b):
        return ' '.join((str(s) for s in b))
    return [f"{a}:'{_key(b)}'".replace('pymordemos.','') for a,b in demo_args]


@pytest.fixture(params=DEMO_ARGS, ids=_demo_ids(DEMO_ARGS))
def demo_args(request):
    _skip_if_no_solver(request.param)
    return request.param


@pytest.fixture(params=THERMALBLOCK_ARGS, ids=_demo_ids(THERMALBLOCK_ARGS))
def thermalblock_args(request):
    _skip_if_no_solver(request.param)
    return request.param


def _test_demo(demo):
    import sys
    sys._called_from_test = True

    def nop(*args, **kwargs):
        pass

    try:
        from matplotlib import pyplot
        pyplot.show = nop
    except ImportError:
        pass
    try:
        import dolfin
        dolfin.plot = nop
    except ImportError:
        pass

    # reset default RandomState
    import pymor.tools.random
    pymor.tools.random._default_random_state = None

    result = None
    try:
        result = demo()
    except QtMissing:
        pytest.xfail("Qt missing")
    except GmshMissing:
        pytest.xfail(f'Gmsh not intalled')
    except MeshioMissing:
        pytest.xfail(f'meshio not intalled')
    finally:
        stop_gui_processes()
        from pymor.parallel.default import _cleanup
        _cleanup()

    return result


def test_demos(demo_args):
    module, args = demo_args
    result = _test_demo(lambda: _run_module(module, args))
    assert result is not None


def test_analyze_pickle1():
    d = mkdtemp()
    try:
        test_demos(('pymordemos.thermalblock', ['--pickle=' + os.path.join(d, 'data'), 2, 2, 2, 10]))
        test_demos(('pymordemos.analyze_pickle',
                   ['histogram', '--error-norm=h1_0_semi', os.path.join(d, 'data_reduced'), 10]))
    finally:
        shutil.rmtree(d)


def test_analyze_pickle2():
    d = mkdtemp()
    try:
        test_demos(('pymordemos.thermalblock', ['--pickle=' + os.path.join(d, 'data'), 2, 2, 2, 10]))
        test_demos(('pymordemos.analyze_pickle',
                   ['histogram', '--detailed=' + os.path.join(d, 'data_detailed'),
                    os.path.join(d, 'data_reduced'), 10]))
    finally:
        shutil.rmtree(d)


def test_analyze_pickle3():
    d = mkdtemp()
    try:
        test_demos(('pymordemos.thermalblock', ['--pickle=' + os.path.join(d, 'data'), 2, 2, 2, 10]))
        test_demos(('pymordemos.analyze_pickle',
                   ['convergence', '--detailed=' + os.path.join(d, 'data_detailed'),
                    '--error-norm=h1_0_semi', os.path.join(d, 'data_reduced'), 10]))
    finally:
        shutil.rmtree(d)


def test_analyze_pickle4():
    d = mkdtemp()
    try:
        test_demos(('pymordemos.thermalblock', ['--pickle=' + os.path.join(d, 'data'), 2, 2, 2, 10]))
        test_demos(('pymordemos.analyze_pickle',
                   ['convergence', '--detailed=' + os.path.join(d, 'data_detailed'),
                    os.path.join(d, 'data_reduced'), 10]))
    finally:
        shutil.rmtree(d)


@pytest.mark.skipif(is_windows_platform(), reason='hangs indefinitely')
@pytest.mark.skipif(is_macos_platform(), reason='spurious JSON Decode errors in Ipython launch')
@pytest.mark.xfail(sys.version_info[0:2]==(3,8), reason='ipyparallel currently broken on python 3.8')
def test_thermalblock_ipython(demo_args):
    if demo_args[0] != 'pymordemos.thermalblock':
        return
    from pymor.tools import mpi
    if mpi.parallel:  # simply running 'ipcluster start' (without any profile) does not seem to work
        return        # when running under mpirun, so we do not test this combination for now
    try:
        test_demos((demo_args[0], ['--ipython-engines=2'] + demo_args[1]))
    finally:
        import time     # there seems to be no way to shutdown the IPyhton cluster s.t. a new
        time.sleep(10)  # cluster can be started directly afterwards, so we have to wait ...


def test_thermalblock_results(thermalblock_args):
    from pymordemos import thermalblock
    results = _test_demo(lambda: thermalblock.main(list(map(str, thermalblock_args[1]))))
    # due to the symmetry of the problem and the random test parameters, the estimated
    # error may change a lot
    # fenics varies more than others between MPI/serial
    first_tolerance = (1e-13, 3.5e-6) if '--fenics' in thermalblock_args[1] else (1e-13, 1e-7)
    check_results('test_thermalblock_results', thermalblock_args[1], results,
                  first_tolerance, 'basis_sizes', 'norms', 'max_norms',
                  (1e-13, 4.), 'errors', 'max_errors', 'rel_errors', 'max_rel_errors',
                  'estimates', 'max_estimates', 'effectivities', 'min_effectivities', 'max_effectivities', 'errors')


def test_burgers_ei_results():
    from pymordemos import burgers_ei
    args = list(map(str, [1, 2, 10, 100, 10, 30]))
    ei_results, greedy_results = _test_demo(lambda: burgers_ei.main(args))
    ei_results['greedy_max_errs'] = greedy_results['max_errs']
    check_results('test_burgers_ei_results', args, ei_results,
                  (1e-13, 1e-7), 'errors', 'triangularity_errors', 'greedy_max_errs')


def test_parabolic_mor_results():
    from pymordemos import parabolic_mor
    args = ['pymor', 'greedy', 5, 20, 3]
    results = _test_demo(lambda: parabolic_mor.main(*args))
    check_results('test_parabolic_mor_results', args, results,
                  (1e-13, 1e-7), 'basis_sizes', 'norms', 'max_norms',
                  (1e-13, 4.), 'errors', 'max_errors', 'rel_errors', 'max_rel_errors',
                  'estimates', 'max_estimates', 'effectivities', 'min_effectivities', 'max_effectivities', 'errors')

if __name__ == "__main__":
    runmodule(filename=__file__)
