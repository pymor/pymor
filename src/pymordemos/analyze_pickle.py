#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Analyze pickled data demo.

Usage:
  analyze_pickle.py histogram [--detailed=DETAILED_DATA] [--error-norm=NORM] REDUCED_DATA SAMPLES
  analyze_pickle.py convergence [--detailed=DETAILED_DATA] [--error-norm=NORM] [--ndim=NDIM] REDUCED_DATA SAMPLES
  analyze_pickle.py (-h | --help)

This demo loads a pickled reduced model, solves for random
parameters, estimates the reduction errors and then visualizes these
estimates. If the detailed model and the reductor are
also provided, the estimated error is visualized in comparison to
the real reduction error.

The needed data files are created by the thermal block demo, by
setting the '--pickle' option.

Arguments:
  REDUCED_DATA  File containing the pickled reduced model.

  SAMPLES       Number of parameter samples to test with.

Options:
  --detailed=DETAILED_DATA  File containing the high-dimensional model
                            and the reductor.

  --error-norm=NORM         Name of norm in which to compute the errors.

  --ndim=NDIM               Number of reduced basis dimensions for which to estimate
                            the error.
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

from pymor.core.pickle import load


def _bins(start, stop, steps=100):
    ''' numpy has a quirk in unreleased master where logspace
    might sometimes not return a 1d array
    '''
    bins = np.logspace(np.log10(start), np.log10(stop), steps)
    if bins.shape == (steps,1):
        bins = bins[:,0]
    return bins


def analyze_pickle_histogram(args):
    args['SAMPLES'] = int(args['SAMPLES'])

    print('Loading reduced model ...')
    rom, parameter_space = load(open(args['REDUCED_DATA'], 'rb'))

    mus = parameter_space.sample_randomly(args['SAMPLES'])
    us = []
    for mu in mus:
        print(f'Solving reduced for {mu} ... ', end='')
        sys.stdout.flush()
        us.append(rom.solve(mu))
        print('done')

    print()

    if hasattr(rom, 'estimate'):
        ests = []
        for u, mu in zip(us, mus):
            print(f'Estimating error for {mu} ... ', end='')
            sys.stdout.flush()
            ests.append(rom.estimate(u, mu=mu))
            print('done')

    if args['--detailed']:
        print('Loading high-dimensional data ...')
        fom, reductor = load(open(args['--detailed'], 'rb'))

        errs = []
        for u, mu in zip(us, mus):
            print(f'Calculating error for {mu} ... ')
            sys.stdout.flush()
            err = fom.solve(mu) - reductor.reconstruct(u)
            if args['--error-norm']:
                errs.append(np.max(getattr(fom, args['--error-norm'] + '_norm')(err)))
            else:
                errs.append(np.max(err.l2_norm()))
            print('done')

        print()

    try:
        plt.style.use('ggplot')
    except AttributeError:
        pass  # plt.style is only available in newer matplotlib versions

    if hasattr(rom, 'estimate') and args['--detailed']:

        # setup axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.02
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        plt.figure(1, figsize=(8, 8))
        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)

        # scatter plot
        total_min = min(np.min(ests), np.min(errs)) * 0.9
        total_max = max(np.max(ests), np.max(errs)) * 1.1
        axScatter.set_xscale('log')
        axScatter.set_yscale('log')
        axScatter.set_xlim([total_min, total_max])
        axScatter.set_ylim([total_min, total_max])
        axScatter.set_xlabel('errors')
        axScatter.set_ylabel('estimates')
        axScatter.plot([total_min, total_max], [total_min, total_max], 'r')
        axScatter.scatter(errs, ests)

        # plot histograms
        x_hist, x_bin_edges = np.histogram(errs, bins=_bins(total_min, total_max))
        axHistx.bar(x_bin_edges[1:], x_hist, width=x_bin_edges[:-1] - x_bin_edges[1:], color='blue')
        y_hist, y_bin_edges = np.histogram(ests, bins=_bins(total_min, total_max))
        axHisty.barh(y_bin_edges[1:], y_hist, height=y_bin_edges[:-1] - y_bin_edges[1:], color='blue')
        axHistx.set_xscale('log')
        axHisty.set_yscale('log')
        axHistx.set_xticklabels([])
        axHisty.set_yticklabels([])
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        axHistx.set_ylim([0, max(np.max(x_hist), np.max(y_hist))])
        axHisty.set_xlim([0, max(np.max(x_hist), np.max(y_hist))])

        plt.show()

    elif hasattr(rom, 'estimate'):

        total_min = np.min(ests) * 0.9
        total_max = np.max(ests) * 1.1

        hist, bin_edges = np.histogram(ests, bins=_bins(total_min, total_max))
        plt.bar(bin_edges[1:], hist, width=bin_edges[:-1] - bin_edges[1:], color='blue')
        plt.xlim([total_min, total_max])
        plt.xscale('log')
        plt.xlabel('estimated error')

        plt.show()

    elif args['--detailed']:

        total_min = np.min(ests) * 0.9
        total_max = np.max(ests) * 1.1

        hist, bin_edges = np.histogram(errs, bins=_bins(total_min, total_max))
        plt.bar(bin_edges[1:], hist, width=bin_edges[:-1] - bin_edges[1:], color='blue')
        plt.xlim([total_min, total_max])
        plt.xscale('log')
        plt.xlabel('error')

        plt.show()

    else:
        raise ValueError('Nothing to plot!')


def analyze_pickle_convergence(args):
    args['SAMPLES'] = int(args['SAMPLES'])

    print('Loading reduced model ...')
    rom, parameter_space = load(open(args['REDUCED_DATA'], 'rb'))

    if not args['--detailed']:
        raise ValueError('High-dimensional data file must be specified.')
    print('Loading high-dimensional data ...')
    fom, reductor = load(open(args['--detailed'], 'rb'))
    fom.enable_caching('disk')

    dim = rom.solution_space.dim
    if args['--ndim']:
        dims = np.linspace(0, dim, args['--ndim'], dtype=np.int)
    else:
        dims = np.arange(dim + 1)

    mus = parameter_space.sample_randomly(args['SAMPLES'])

    ESTS = []
    ERRS = []
    T_SOLVES = []
    T_ESTS = []
    for N in dims:
        rom = reductor.reduce(N)
        print(f'N = {N:3} ', end='')
        us = []
        print('solve ', end='')
        sys.stdout.flush()
        start = time.time()
        for mu in mus:
            us.append(rom.solve(mu))
        T_SOLVES.append((time.time() - start) * 1000. / len(mus))

        print('estimate ', end='')
        sys.stdout.flush()
        if hasattr(rom, 'estimate'):
            ests = []
            start = time.time()
            for u, mu in zip(us, mus):
                # print('e', end='')
                # sys.stdout.flush()
                ests.append(rom.estimate(u, mu=mu))
            ESTS.append(max(ests))
            T_ESTS.append((time.time() - start) * 1000. / len(mus))

        print('errors', end='')
        sys.stdout.flush()
        errs = []
        for u, mu in zip(us, mus):
            err = fom.solve(mu) - reductor.reconstruct(u)
            if args['--error-norm']:
                errs.append(np.max(getattr(fom, args['--error-norm'] + '_norm')(err)))
            else:
                errs.append(np.max(err.l2_norm()))
        ERRS.append(max(errs))

        print()

    print()

    try:
        plt.style.use('ggplot')
    except AttributeError:
        pass  # plt.style is only available in newer matplotlib versions

    plt.subplot(1, 2, 1)
    if hasattr(rom, 'estimate'):
        plt.semilogy(dims, ESTS, label='max. estimate')
    plt.semilogy(dims, ERRS, label='max. error')
    plt.xlabel('dimension')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dims, T_SOLVES, label='avg. solve time')
    if hasattr(rom, 'estimate'):
        plt.plot(dims, T_ESTS, label='avg. estimate time')
    plt.xlabel('dimension')
    plt.ylabel('milliseconds')
    plt.legend()

    plt.show()


def analyze_pickle_demo(args):
    if args['histogram']:
        analyze_pickle_histogram(args)
    else:
        analyze_pickle_convergence(args)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    analyze_pickle_demo(args)
