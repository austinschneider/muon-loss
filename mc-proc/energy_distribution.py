import numpy as np
import matplotlib
matplotlib.use('Agg')
from lmfit import minimize, Parameters, Parameter, report_fit
import matplotlib.pyplot as plt 
import scipy.optimize

from icecube import dataclasses

import os
import pickle
import cPickle
import glob
import argparse
import itertools
import re
import operator as op

import muon_loss_common as mlc

def add_point(hb, bins, loss_tuples, weight, cps, mu, nu, run_id, event_id):
    # Make some cuts
    #if(mu[mlc.muon_p_energy] < 5000):
    #    return False

    # Values to accumulate
    point_E = np.zeros((len(bins),0))
    point_weight = np.zeros((len(bins),0))

    # We should always have 5 checkpoints
    if len(cps) != 5:
        raise ValueError("There are %d checkpoints instead of 5." % len(cps))
    
    min_range, max_range = mlc.get_track_range(cps)        
    new_cps = mlc.get_valid_checkpoints(cps)

    track_cps = cps[1:-1]

    starts_outside_simvol = track_cps[0][0] > 0
    #if(starts_outside_simvol and track_cps[0][0] < 5000):
    #    return False

    # Sanity check on the segment of track we are considering
    if max_range - min_range > 10000:
        print("Range greater than 10000m!")

    if max_range - min_range < max(bins):
        return False

    # Convert to tuple for memoization
    loss_tuples = tuple(loss_tuples)

    for i,hist in enumerate(hb.hists):
        x = bins[i]
        x += min_range
        if(x > max_range):
            continue
        else:
            # Total losses
            E_at_x = mlc.get_energy(x, new_cps, loss_tuples)
        hist.add([E_at_x], [1], [weight])

    return True

def make_hist(bins, min_E=-2, max_E=5, bins_per_decade=10):
    E_bins = np.logspace(min_E,max_E,(max_E-min_E)*bins_per_decade+1)
    hists = []
    for i in xrange(len(bins)):
        hists.append(mlc.histogram(E_bins, store_data=False))
    return mlc.histogram_bundle(hists)

def plot(hb, bins, plotdir, split=False):
    cs=matplotlib.cm.jet
    fig, ax = plt.subplots(1)
    max_y = -np.inf
    min_y = np.inf
    for i,hist in enumerate(hb.hists):
        y = hist.get_y()
        w = hist.get_w()
        y = y*w / sum(w)
        x = (hist.bins[:-1] + hist.bins[1:]) / 2.0
        data, edges, patches = plt.hist(x, bins=hist.bins, weights=y, label='Energy After %dm' % bins[i], color=cs((bins[i] - bins[0]) / bins[-1]), histtype='step')
        min_y = min(min_y, min(data[data > 0]))
        max_y = max(max_y, max(data[data > 0]))

    
    max_y = 10**np.ceil(np.log10(max_y))
    min_y = 10**np.floor(np.log10(min_y))
    plt.legend(loc=2)
    plt.xlabel('Muon Energy (GeV)')
    plt.ylabel('Number of Events')
    plt.ylim((min_y,max_y))
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(plotdir + 'energy_distribution_' + ('%d_to_%d_%dbins' % (min(bins), max(bins), len(bins)-1)) + '.png')
    plt.clf()
    plt.close(fig)

