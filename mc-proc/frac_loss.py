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

def fit_for_b(bins, x, y, error=None):
    """
    Fit a fractional energy loss histogram for the parameter b
    """
    magic_ice_const = 0.917

    outer_bounds = (min(bins), max(bins)) # Beginning and ending position
    select_bounds = np.vectorize(lambda xx: mlc.get_bounding_elements(xx, bins)) # Returns the bin edges on either side of a point
    E_diff = lambda xx, b: np.exp(-b*xx[1]) - np.exp(-b*xx[0]) # Proportional to Delta E (comes from -dE/dx=a+b*E)
    fit_func = lambda xx, b: E_diff(select_bounds(xx), b*magic_ice_const) / E_diff(outer_bounds, b*magic_ice_const) # We are fitting to the ratio of differences
        
    params = Parameters()
    params.add('b', value=0.4*10**(-3)) # Add b as a fit parameter
    if error is not None:
        l_fit_func = lambda params, x, data: np.sqrt((fit_func(x, params['b']) - data)**2 / error**2)
    else:
        l_fit_func = lambda params, x, data: fit_func(x, params['b']) - data

    result = minimize(l_fit_func, params, args=(x, y)) 

    b = result.params['b'].value

    if b == 0.4*10**(-3):
        print fit 
        print x
        print y
        print fit_func(x, 0.36*10**(-3))
        print w
        raise ValueError("Fit doesn't make sense.")

    return b

def add_point(hist, bins, loss_tuples, weight, cps, mu, nu, run_id, event_id):
    """
    Add points to histogram for fractional energy loss plot
    """

    hist, ratio_hist = hist.hists

    # Make some cuts
    if(mu[mlc.muon_p_energy] < 1000):
    #    print 'Energy cut'
        return False

    # Values to accumulate
    point_x = []
    point_dE = []
    point_frac_dE = []
    point_E_at_x1 = []
    point_E_at_x2 = []
    point_weight = []

    # We should always have 5 checkpoints
    if len(cps) != 5:
        raise ValueError("There are %d checkpoints instead of 5." % len(cps))
    
    min_range, max_range = mlc.get_track_range(cps)
    new_cps = mlc.get_valid_checkpoints(cps)

    track_cps = cps[1:-1]

    starts_outside_simvol = track_cps[0][0] > 0
    if(starts_outside_simvol and track_cps[0][0] < 1000):
    #    print 'Energy cut'
        return False

    # Sanity check on the segment of track we are considering
    if max_range - min_range > 10000:
        print("Range greater than 10000m!")

    # Skip tracks that do not cover the full range
    if max_range - min_range < max(hist.bins):
        return False

    # Convert to tuple for memoization
    loss_tuples = tuple(loss_tuples)

    # Calculate the fractional energy loss between each bin edge
    for x1,x2 in itertools.izip(hist.bins[:-1], hist.bins[1:]):
        x1 += min_range
        x2 += min_range
        if(x1 > max_range):
            continue
        else:
            if(x2 > max_range):
                x2 = max_range
            # Total losses
            E_at_x1 = mlc.get_energy(x1, new_cps, loss_tuples)
            E_at_x2 = mlc.get_energy(x2, new_cps, loss_tuples)
            x, dE = (x1, abs((E_at_x1 - E_at_x2)))
        point_x.append(x)
        point_dE.append(dE)
        point_E_at_x1.append(E_at_x1)
        point_E_at_x2.append(E_at_x2)
        point_weight.append(weight)
    total_dE = sum(point_dE)
    if(total_dE == 0):
        return False
    point_frac_dE = [dE / total_dE for dE in point_dE]
    point_x = [x - min_range for x in point_x]

    # Add relavent points to histogram
    hist.add(point_x, [point_frac_dE, point_dE, [total_dE]*len(point_dE), point_E_at_x1, point_E_at_x2, [point_E_at_x1[0]]*len(point_dE), [point_E_at_x2[-1]]*len(point_dE)], point_weight)
    ratio_hist.add([point_frac_dE[0] / point_frac_dE[-1]], [1], [weight])

    return True

def make_hist(bins):
    frac_loss_hist = mlc.histogram_nd(7, bins, store_data=False)
    min_power = -3
    max_power = 3
    ratio_distribution_hist = mlc.histogram_nd(1, np.logspace(min_power, max_power, (max_power - min_power)*16+1), store_data=False)
    
    return mlc.histogram_bundle([frac_loss_hist, ratio_distribution_hist])

def plot_frac_loss(hist, bins, plotdir, split=False):
    split = int(split)

    #colors = ['b', 'g', 'm', 'c']
    color = 'b'
    label = 'MC Total Losses'
    
    #fig = plt.figure()
    fig, ax = plt.subplots(1)
    
    x = (hist.bins[:-1] + hist.bins[1:]) / 2 
    dEdeltaE, dE, deltaE, E_at_x1, E_at_x2, E_at_x3, E_at_x4 = hist.get_y()

    dEdeltaE_e, dE_e, deltaE_e, Ex1_e, Ex2_e, Ex3_e, Ex4_e = hist.get_y_stddev_of_mean()

    if split == 2:
        dE = abs(E_at_x1 - E_at_x2)
        deltaE = abs(E_at_x3 - E_at_x4)

        dE_e = np.sqrt(Ex1_e**2.0 + Ex2_e**2.0)
        deltaE_e = np.sqrt(Ex3_e**2.0 + Ex4_e**2.0)

    if split:
        deltaE_e[deltaE == 0] = 0 
        dE_e[dE == 0] = 0 

        deltaE[deltaE == 0] = 1 
        y = dE / deltaE

        dE[dE == 0] = 1 
        error_of_mean_by_bin = abs(dEdeltaE) * np.sqrt((dE_e/abs(dE))**2 + (deltaE_e/abs(deltaE))**2)
    else:
        y = dEdeltaE
        error_of_mean_by_bin = dEdeltaE_e

    alt_err = np.where(y < error_of_mean_by_bin, 0, error_of_mean_by_bin)

    plt.errorbar(x, y, fmt=color+'.', label=label, yerr=[alt_err, error_of_mean_by_bin])
    try:
        plt.hist(x, bins=hist.bins, weights=y, color=color, histtype='step')
    except:
        print(label)
        print("x: ", x)
        print("y: ", y)
        print hist.get_y()
        print hist.get_w()
        raise

    magic_ice_const = 0.917

    x2 = max(hist.bins)
    dxs = np.unique(hist.bins[1:].astype(int) - hist.bins[:-1].astype(int))
    select_dx = np.vectorize(lambda x: dxs[next(itertools.dropwhile(lambda xx: xx[1] > x, enumerate(hist.bins)))[0]])
    c_func = lambda b, dx, x2: np.log(b*dx/(1-np.exp(-b*x2)))

    fit_func = lambda x, b: np.exp(-b*x*magic_ice_const+c_func(b*magic_ice_const, select_dx(x), x2))
    fit = scipy.optimize.curve_fit(fit_func, x, y, [4.68*10**(-4)], error_of_mean_by_bin)
    print('Exponential fit: y=Exp[%.9f * x + %f]' % (-fit[0][0], c_func(fit[0][0], sum(dxs)/len(dxs), x2)))                                                                                             

    print('L=%f' % ((fit[0][0])**(-1)))
    
    b = fit_for_b(hist.bins, x, y, error_of_mean_by_bin)

    L = 1.0/b

    print('L=%f' % (L))

    plt.plot(x, fit_func(x, b), 'r')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.70, 0.1, "L=%.03f (m)" % L, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)

    plt.legend(loc=3)
    plt.xlabel('Track Distance (m)')
    plt.ylabel('Fractional Energy Loss')
    plt.title(label)
    plt.savefig(plotdir + 'fractional_energy_loss_' + re.sub(' ', '_', label.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}')) + ('%d_to_%d_%dbins' % (min(bins), max(bins), len(bins)-1)) + '_split_' + str(split) + '.eps', format='eps', dpi=1000)
    plt.clf()
    plt.close(fig)

plot_split_0 = lambda *x: plot_frac_loss(*x, split=False)
plot_split_1 = lambda *x: plot_frac_loss(*x, split=True)
plot_split_2 = lambda *x: plot_frac_loss(*x, split=2)

def plot_ratio(hist, bins, plotdir):
    fig, ax = plt.subplots(1)

    y = hist.get_y()[0]
    w = hist.get_w()
    y = y*w / sum(w)
    x = (hist.bins[:-1] + hist.bins[1:]) / 2.0

    min_y = np.log10(min(y[y > 0]))
    max_y = np.log10(max(y[y > 0]))

    select = lambda x, fx: fx if abs(fx - x) < 0.5 else x
    min_y = 10.0**select(min_y, np.floor(min_y))
    max_y = 10.0**select(max_y, np.ceil(max_y))

    color = 'b'

    plt.hist(x, bins=hist.bins, weights=y, color=color, histtype='step', label=('(%dm-%dm) / (%dm-%dm)' % (bins[0], bins[1], bins[-2], bins[-1])))
    plt.xlabel('Ratio of Fractional Energy Losses')
    plt.ylabel('Number of Events')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((min_y, max_y))
    plt.legend(loc=3)
    plt.savefig(plotdir + 'fractional_energy_loss_ratio_' + ('%d-%d_%d-%d' % (bins[0], bins[1], bins[-2], bins[-1])) + '.eps', format='eps', dpi=1000)
    plt.clf()
    plt.close(fig)


def plot(hb, bins, plotdir):
    hist, ratio_hist = hb.hists
    plot_split_0(hist, bins, plotdir)
    plot_split_1(hist, bins, plotdir)
    plot_split_2(hist, bins, plotdir)
    plot_ratio(ratio_hist, bins, plotdir)
