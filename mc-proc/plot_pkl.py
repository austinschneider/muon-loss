#!/usr/bin/env python
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

# Set up argument parser
# Output directory stores nothing
# Max distance is the maximum distance considered for energy loss
# Bin size is the size of the distance bins in meters
parser = argparse.ArgumentParser(description='Proccess filenames')
parser.add_argument('indirs', metavar='indirs',  nargs='*')
parser.add_argument('-o', '--outfile', default='')
parser.add_argument('-i', '--infile', default='')
parser.add_argument('-p', '--plotdir', default='')

parser.add_argument('--aggregate', dest='aggregate', action='store_true')
parser.add_argument('--no-aggregate', dest='aggregate', action='store_false')
parser.set_defaults(aggregate=True)

parser.add_argument('-r', '--range', default='')

args = parser.parse_args()
indirs = args.indirs
outfile = args.outfile.translate(None, '"\'')
infile = args.infile.translate(None, '"\'')
plotdir = args.plotdir.translate(None, '"\'')
aggregate = args.aggregate
range = args.range.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}_ ')
if range == '':
    range = None
else:
    range = [int(i) for i in range.split('-', 1)]

# Define fucntion to get item from list or tuple
_get0 = op.itemgetter(0)
_get1 = op.itemgetter(1)

def fit_for_b_(bins, stuff):

        w = stuff['w']
        w2 = stuff['w2']

        w[w == 0] = 1
        w2[w2 == 0] = 1

        #x = stuff['x'] / w
        x = (bins[:-1] + bins[1:]) / 2
        y = stuff['y'][0] / w
        x2 = stuff['x2'] / w
        y2 = stuff['y2'][0] / w

        return fit_for_b(bins, x, y, None)

def fit_for_b(bins, x, y, error=None):

        magic_ice_const = 0.917

        outer_bounds = (min(bins), max(bins))
        select_bounds = np.vectorize(lambda xx: mlc.get_bounding_elements(xx, bins))
        E_diff = lambda xx, b: np.exp(-b*xx[1]) - np.exp(-b*xx[0])
        fit_func = lambda xx, b: E_diff(select_bounds(xx), b*magic_ice_const) / E_diff(outer_bounds, b*magic_ice_const)
       
        params = Parameters()
        params.add('b', value=0.4*10**(-3))
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

def plot(hists, dist_bins, points_labels, plotdir, split=False):

    colors = ['b', 'g', 'm', 'c']
    
    #fig = plt.figure()
    fig, ax = plt.subplots(1)

    for hist,label,color in itertools.izip(hists, points_labels, colors):
        print(label)
        
        #x = hist.get_x()
        x = (hist.bins[:-1] + hist.bins[1:]) / 2
        dEdeltaE, dE, deltaE = hist.get_y()

        dEdeltaE_e, dE_e, deltaE_e = hist.get_y_stddev_of_mean()

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
            plt.hist(x, bins=dist_bins, weights=y, color=color, histtype='step')
        except:
            print(label)
            print("x: ", x)
            print("y: ", y)
            raise

        magic_ice_const = 0.917

        x2 = max(hist.bins)
        dxs = np.unique(hist.bins[1:].astype(int) - hist.bins[:-1].astype(int))
        select_dx = np.vectorize(lambda x: dxs[next(itertools.dropwhile(lambda xx: xx[1] > x, enumerate(hist.bins)))[0]])
        c_func = lambda b, dx, x2: np.log(b*dx/(1-np.exp(-b*x2)))

        fit_func = lambda x, b: np.exp(-b*x*magic_ice_const+c_func(b*magic_ice_const, select_dx(x), x2))
        #fit_func = lambda x, b, a: x*b*0.917+a*0.917
        # Fit the histogram datapoints to an exponential
        #fit = scipy.optimize.curve_fit(fit_func, x, y, [-4.68*10**(-4), -2.15], error_of_mean_by_bin)
        fit = scipy.optimize.curve_fit(fit_func, x, y, [4.68*10**(-4)], error_of_mean_by_bin)
        #print('Exponential fit: y=Exp[%f * x + %f]' % (fit[0][0]*0.917, fit[0][1]*0.917))
        print('Exponential fit: y=Exp[%.9f * x + %f]' % (-fit[0][0], c_func(fit[0][0], sum(dxs)/len(dxs), x2)))

        print('L=%f' % ((fit[0][0])**(-1)))

        #outer_bounds = (min(hist.bins), max(hist.bins))
        #select_bounds = np.vectorize(lambda xx: mlc.get_bounding_elements(xx, hist.bins))
        #E_diff = lambda xx, b: np.exp(-b*xx[1]) - np.exp(-b*xx[0])
        #fit_func = lambda xx, b: E_diff(select_bounds(xx), b*magic_ice_const) / E_diff(outer_bounds, b*magic_ice_const)
        
        #fit = scipy.optimize.curve_fit(fit_func, x, y, [.36*10**(-3)], error_of_mean_by_bin)

        #print('L=%f' % ((fit[0][0])**(-1)))

        b = fit_for_b(dist_bins, x, y, error_of_mean_by_bin)

        L = 1.0/b

        print('L=%f' % (L))

        plt.plot(x, fit_func(x, b), 'r')
    
        #line = lambda x: 0.259*0.917+0.363*10**(-3)*0.917*x
        #plt.plot(x,line(x), 'r.', linewidth=2.0, label='Expected total losses')
        #plt.axis([1, 10**6, 10**(-4), 10**4])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.70, 0.1, "L=%.03f (m)" % L, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)

        plt.legend(loc=3)
        plt.xlabel('Track Distance (m)')
        plt.ylabel('Fractional Energy Loss')
        plt.title(label)
        #plt.xscale('log')
        #plt.yscale('log')
        if split:
            plt.savefig(plotdir + 'fractional_energy_loss_' + re.sub(' ', '_', label.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}')) + '_split.png')
        else:
            plt.savefig(plotdir + 'fractional_energy_loss_' + re.sub(' ', '_', label.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}')) + '.png')
        plt.clf()
        plt.close(fig)
    #plt.show()

bs = None

def add_points(losses, weights, checkpoints, mu_info, points_functions, hists, E_bins, sample_d = 10, sample_E = 10):
    """
    Add points to the histogram for each muon with appropriate weighting
    """
    global bs
    if not bs:
        bs = np.zeros((len(hists), 0)).tolist()
    E_bins = sorted(E_bins)[:]
    n_muons = 0
    
    for cps,loss_tuples,mu,weight in itertools.izip(checkpoints, losses, mu_info, weights):
        # Make some cuts
        if(mu[mlc.muon_p_energy] < 5000):
            continue
        
        #if(np.sqrt(mu[mlc.muon_p_pos_x]**2 + mu[mlc.muon_p_pos_y]**2 + mu[mlc.muon_p_pos_z]**2) > 500):
        #    continue

        # Values to accumulate
        point_x = np.zeros((len(hists), 0)).tolist()
        point_dE = np.zeros((len(hists), 0)).tolist()
        point_frac_dE = np.zeros((len(hists), 0)).tolist()
        point_weight = np.zeros((len(hists), 0)).tolist()

        # We should always have 5 checkpoints
        if len(cps) != 5:
            raise ValueError("There are %d checkpoints instead of 5." % len(cps))
        
        min_range, max_range = mlc.get_track_range(cps)        
        new_cps = mlc.get_valid_checkpoints(cps)

        track_cps = cps[1:-1]
    
        starts_outside_simvol = track_cps[0][0] > 0
        if(starts_outside_simvol and track_cps[0][0] < 5000):
            continue
        #if(not starts_outside_simvol and mu[mlc.muon_p_energy] > 6000):
        #    continue
        #if(starts_outside_simvol and track_cps[0][0] > 6000):
        #    continue

        # Sanity check on the segment of track we are considering
        if max_range - min_range > 10000:
            print("Range greater than 10000m!")

        if max_range - min_range < max(E_bins):
            continue

        #if mlc.get_energy(min_range, new_cps, loss_tuples) - mlc.get_energy(max_range, new_cps, loss_tuples) < 500:
        #    continue

        #if get_energy(min_range + max(E_bins), new_cps, loss_tuples, has_sum=True, inclusive=False) < 1000:
        #    continue

        # Convert to tuple for memoization
        loss_tuples = tuple(loss_tuples)

        for i in xrange(len(hists)):
            for x1,x2 in itertools.izip(hists[i].bins[:-1], hists[i].bins[1:]):
                x1 += min_range
                x2 += min_range
                if(x1 > max_range):
                    continue
                else:
                    if(x2 > max_range):
                        x2 = max_range
                    x, dE = points_functions[i]((x1, x2, new_cps, loss_tuples))
                point_x[i].append(x)
                point_dE[i].append(dE)
                point_weight[i].append(weight)
            total_dE = sum(point_dE[i])
            if(total_dE == 0):
                continue
            point_frac_dE[i] = [dE / total_dE for dE in point_dE[i]]
            point_x[i] = [x - min_range for x in point_x[i]]

            hists[i].add(point_x[i], [point_frac_dE[i], point_dE[i], [total_dE]*len(point_dE[i])], point_weight[i])

        n_muons += 1

        if not n_muons % 1000:
            print("%d muons in file" % n_muons)

        # Clear the memoization dictionary for get_loss_info since we are done with the muon
        mlc.get_loss_info.__self__.clear()
        
    return n_muons

def get_hists_from_dirs(indirs, E_bins, points_functions, range = None):
    """
    Compute histograms from pickle contained the list of input directories
    """
    # Get list of input pickle files from all input directories
    infiles = [file for indir in indirs for file in glob.glob('%s*.pkl' % indir)]
    infiles.sort()

    if range is not None:
        if len(range) == 2:
            infiles = infiles[range[0]:range[1]]
        elif len(range) == 1:
            infiles = [infiles[range[0]]]

    n_muons = 0

    # Create a histogram object for each function
    hists = []
    for i in points_functions:
        hists.append(mlc.histogram_nd(3, E_bins, store_data=False))

    # Loop over input files
    for infile in infiles:
        if os.stat(infile).st_size == 0:
            continue
        print(infile)

        # Load losses for each muon
        inpickle = open(infile, 'rb')
        losses = cPickle.load(inpickle)

        # Load weights for each muon
        weights = cPickle.load(inpickle)

        # Load track energy checkpoints for each muon
        checkpoints = cPickle.load(inpickle)
        
        # Load muon and neutrino information
        mu_info = cPickle.load(inpickle)
        #nu_info = np.array(pickle.load(inpickle))

        # Preemptively calculate the sum of losses since the last valid checkpoint and append the information to the loss
        # Offers drastic performance improvement
        for i in xrange(len(losses)):
            losses[i] = mlc.add_loss_sum(losses[i], mlc.get_valid_checkpoints(checkpoints[i]))
        
        # Add points to the histograms using the points_functions
        n_muons += add_points(losses, weights, checkpoints, mu_info, points_functions, hists, E_bins)

        # Get rid of memory heavy references
        losses = None
        weights = None
        checkpoints = None
        mu_info = None
        nu_info = None

        print("Have %d muons!" % n_muons)

        inpickle.close()

    return hists

exclude = ['deltaE']
exclude = set(mlc.get_particle_number(exclude))

def get_total_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    E_at_x1 = mlc.get_energy(x1, cps, loss_tuples)
    E_at_x2 = mlc.get_energy(x2, cps, loss_tuples)
    return (x1, abs((E_at_x1 - E_at_x2)))
   
def get_stoch_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    return (x1, abs(sum([loss[0] for loss in loss_tuples if loss[2] not in exclude and loss[1] > x1 and loss[1] <= x2])))

def get_mc_stoch_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    return (x1, abs(sum([loss[0] for loss in loss_tuples if loss[1] > x1 and loss[1] <= x2]) / (x2 - x1)))

def get_cont_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    E_at_x1 = mlc.get_energy(x1, cps, loss_tuples)
    E_at_x2 = mlc.get_energy(x2, cps, loss_tuples)
    return (x1, (abs((E_at_x1 - E_at_x2)) - abs(sum([loss[0] for loss in loss_tuples if loss[1] > x1 and loss[1] <= x2]))) / (x2 - x1))

#points_functions = [get_total_losses_point, get_stoch_losses_point, get_mc_stoch_losses_point, get_cont_losses_point]
#points_labels = ['MC total losses', 'MC stochastic losses', 'MC stochastic losses w/ ionization', 'MC continuous losses']

points_functions = [get_total_losses_point]
points_labels = ['MC total losses']

dist_bins = np.linspace(0, 600, 10+1)

if plotdir != '' or outfile != '':
    if len(infile) == 0 and len(indirs) > 0:
        if aggregate:
            hists, dist_bins = mlc.aggregate_info_from_dirs(indirs)
        else:
            hists = get_hists_from_dirs(indirs, dist_bins, points_functions, range)
        if outfile != '':
            mlc.save_info_to_file(outfile, hists, dist_bins)
    elif len(infile) > 0:
        hists, E_bins = mlc.get_info_from_file(infile)
    if outfile != '':
        mlc.save_info_to_file(outfile, hists, dist_bins)
    if plotdir != '':
        plot(hists, dist_bins, points_labels, plotdir, split=False)
