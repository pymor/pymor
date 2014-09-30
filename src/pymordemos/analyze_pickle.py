#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Analyze pickled data demo.

Usage:
  analyze_pickly.py [-h] [--detailed=DETAILED_DATA] [--error-norm=NORM] [--help]
                    REDUCED_DATA SAMPLES

This demo loads a pickled reduced discretization, solves for random
parameters, estimates the reduction errors and then visualizes these
estimates. If the detailed discretization and the reconstructor are
also provided, the estimated error is visualized in comparison to
the real reduction error.

The needed data files are created by the thermal block demo, by
setting the '--pickle' option.

Arguments:
  REDUCED_DATA  File containing the pickled reduced discretization.

  SAMPLES       Number of samples to test with.

Options:
  --detailed=DETAILED_DATA  File containing the high-dimensional discretization
                            and the reconstructor.

  --error-norm=NORM         Name of norm in which to compute the errors.
"""

from __future__ import absolute_import, division, print_function

from pymor.core.defaults import set_defaults

import sys
import math as m
import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

import pymor.core as core
from pymor.core.pickle import load
from pymor.discretizers import discretize_elliptic_cg
from pymor.reductors import reduce_to_subbasis
from pymor.reductors.linear import reduce_stationary_affine_linear
core.set_log_levels({'pymor.algorithms': 'INFO',
                     'pymor.discretizations': 'INFO',
                     'pymor.la': 'INFO'})


def analyze_pickle_demo(args):
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
        plt.figure(1, figsize=(8,8))
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



if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    analyze_pickle_demo(args)
