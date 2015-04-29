#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Analyze pickled data demo.

Usage:
  analyze_pickle.py histogram [--detailed=DETAILED_DATA] [--error-norm=NORM] REDUCED_DATA SAMPLES
  analyze_pickle.py convergence [--detailed=DETAILED_DATA] [--error-norm=NORM] [--ndim=NDIM] REDUCED_DATA SAMPLES
  analyze_pickle.py (-h | --help)

This demo loads a pickled reduced discretization, solves for random
parameters, estimates the reduction errors and then visualizes these
estimates. If the detailed discretization and the reconstructor are
also provided, the estimated error is visualized in comparison to
the real reduction error.

The needed data files are created by the thermal block demo, by
setting the '--pickle' option.

Arguments:
  REDUCED_DATA  File containing the pickled reduced discretization.

  SAMPLES       Number of parameter samples to test with.

Options:
  --detailed=DETAILED_DATA  File containing the high-dimensional discretization
                            and the reconstructor.

  --error-norm=NORM         Name of norm in which to compute the errors.

  --ndim=NDIM               Number of reduced basis dimensions for which to estimate
                            the error.
"""

from __future__ import absolute_import, division, print_function

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

from pymor.core.pickle import load
from pymor.reductors.basic import reduce_to_subbasis


def analyze_pickle_histogram(args):
    args['SAMPLES'] = int(args['SAMPLES'])

    print('Loading reduced discretization ...')
    rb_discretization = load(open(args['REDUCED_DATA']))

    mus = list(rb_discretization.parameter_space.sample_randomly(args['SAMPLES']))
    us = []
    for mu in mus:
        print('Solving reduced for {} ... '.format(mu), end='')
        sys.stdout.flush()
        us.append(rb_discretization.solve(mu))
        print('done')

    print()

    if hasattr(rb_discretization, 'estimate'):
        ests = []
        for u, mu in zip(us, mus):
            print('Estimating error for {} ... '.format(mu), end='')
            sys.stdout.flush()
            ests.append(rb_discretization.estimate(u, mu=mu))
            print('done')

    if args['--detailed']:
        print('Loading high-dimensional data ...')
        discretization, reconstructor = load(open(args['--detailed']))

        errs = []
        for u, mu in zip(us, mus):
            print('Calculating error for {} ... '.format(mu))
            sys.stdout.flush()
            err = discretization.solve(mu) - reconstructor.reconstruct(u)
            if args['--error-norm']:
                errs.append(np.max(getattr(discretization, args['--error-norm'] + '_norm')(err)))
            else:
                errs.append(np.max(err.l2_norm()))
            print('done')

        print()

    try:
        plt.style.use('ggplot')
    except AttributeError:
        pass  # plt.style is only available in newer matplotlib versions

    if hasattr(rb_discretization, 'estimate') and args['--detailed']:

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
        total_min = min(min(ests), min(errs)) * 0.9
        total_max = max(max(ests), max(errs)) * 1.1
        axScatter.set_xscale('log')
        axScatter.set_yscale('log')
        axScatter.set_xlim([total_min, total_max])
        axScatter.set_ylim([total_min, total_max])
        axScatter.set_xlabel('errors')
        axScatter.set_ylabel('estimates')
        axScatter.plot([total_min, total_max], [total_min, total_max], 'r')
        axScatter.scatter(errs, ests)

        # plot histograms
        x_hist, x_bin_edges = np.histogram(errs, bins=np.logspace(np.log10(total_min), np.log10(total_max), 100))
        axHistx.bar(x_bin_edges[1:], x_hist, width=x_bin_edges[:-1] - x_bin_edges[1:], color='blue')
        y_hist, y_bin_edges = np.histogram(ests, bins=np.logspace(np.log10(total_min), np.log10(total_max), 100))
        axHisty.barh(y_bin_edges[1:], y_hist, height=y_bin_edges[:-1] - y_bin_edges[1:], color='blue')
        axHistx.set_xscale('log')
        axHisty.set_yscale('log')
        axHistx.set_xticklabels([])
        axHisty.set_yticklabels([])
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        axHistx.set_ylim([0, max(max(x_hist), max(y_hist))])
        axHisty.set_xlim([0, max(max(x_hist), max(y_hist))])

        plt.show()

    elif hasattr(rb_discretization, 'estimate'):

        total_min = min(ests) * 0.9
        total_max = max(ests) * 1.1

        hist, bin_edges = np.histogram(ests, bins=np.logspace(np.log10(total_min), np.log10(total_max), 100))
        plt.bar(bin_edges[1:], hist, width=bin_edges[:-1] - bin_edges[1:], color='blue')
        plt.xlim([total_min, total_max])
        plt.xscale('log')
        plt.xlabel('estimated error')

        plt.show()

    elif args['--detailed']:

        total_min = min(ests) * 0.9
        total_max = max(ests) * 1.1

        hist, bin_edges = np.histogram(errs, bins=np.logspace(np.log10(total_min), np.log10(total_max), 100))
        plt.bar(bin_edges[1:], hist, width=bin_edges[:-1] - bin_edges[1:], color='blue')
        plt.xlim([total_min, total_max])
        plt.xscale('log')
        plt.xlabel('error')

        plt.show()

    else:
        raise ValueError('Nothing to plot!')


def analyze_pickle_convergence(args):
    args['SAMPLES'] = int(args['SAMPLES'])

    print('Loading reduced discretization ...')
    rb_discretization = load(open(args['REDUCED_DATA']))

    if args['--detailed']:
        print('Loading high-dimensional data ...')
        discretization, reconstructor = load(open(args['--detailed']))

    if not hasattr(rb_discretization, 'estimate') and not args['--detailed']:
        raise ValueError('Nothing to do! (Neither estimates nor true error can be computed.)')

    dim = rb_discretization.solution_space.dim
    if args['--ndim']:
        dims = np.linspace(0, dim, args['--ndim'], dtype=np.int)
    else:
        dims = np.arange(dim + 1)

    mus = list(rb_discretization.parameter_space.sample_randomly(args['SAMPLES']))

    ESTS = []
    ERRS = []
    T_SOLVES = []
    T_ESTS = []
    for N in dims:
        rd, rc, _ = reduce_to_subbasis(rb_discretization, N)
        print('N = {:3} '.format(N), end='')
        us = []
        print('solve ', end='')
        sys.stdout.flush()
        start = time.time()
        for mu in mus:
            us.append(rd.solve(mu))
        T_SOLVES.append((time.time() - start) * 1000. / len(mus))

        print('estimate ', end='')
        sys.stdout.flush()
        if hasattr(rb_discretization, 'estimate'):
            ests = []
            start = time.time()
            for u, mu in zip(us, mus):
                # print('e', end='')
                # sys.stdout.flush()
                ests.append(rd.estimate(u, mu=mu))
            ESTS.append(max(ests))
            T_ESTS.append((time.time() - start) * 1000. / len(mus))

        if args['--detailed']:
            print('errors', end='')
            sys.stdout.flush()
            errs = []
            for u, mu in zip(us, mus):
                err = discretization.solve(mu) - reconstructor.reconstruct(rc.reconstruct(u))
                if args['--error-norm']:
                    errs.append(np.max(getattr(discretization, args['--error-norm'] + '_norm')(err)))
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
    if hasattr(rb_discretization, 'estimate'):
        plt.semilogy(dims, ESTS, label='max. estimate')
    if args['--detailed']:
        plt.semilogy(dims, ERRS, label='max. error')
    plt.xlabel('dimension')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dims, T_SOLVES, label='avg. solve time')
    if hasattr(rb_discretization, 'estimate'):
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
