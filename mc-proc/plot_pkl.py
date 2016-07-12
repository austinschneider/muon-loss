import numpy as np
import matplotlib
matplotlib.use('Agg')
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

args = parser.parse_args()
indirs = args.indirs
outfile = args.outfile
infile = args.infile
plotdir = args.plotdir

# Define fucntion to get item from list or tuple
_get0 = op.itemgetter(0)
_get1 = op.itemgetter(1)

def plot(hists, dist_bins, points_labels, plotdir):

    colors = ['b', 'g', 'm', 'c']
    
    fig = plt.figure()

    for hist,label,color in itertools.izip(hists, points_labels, colors):
        print(label)
        
        #x = hist.get_x()
        x = (hist.bins[:-1] + hist.bins[1:]) / 2
        y = hist.get_y()

        error_of_mean_by_bin = hist.get_y_stddev_of_mean()

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

        fit_func = lambda x, b: np.exp(-b*x+c_func(b, select_dx(x), x2))
        #fit_func = lambda x, b, a: np.exp(x*b*0.917+a*0.917)
        #fit_func = lambda x, b, a: x*b*0.917+a*0.917
        # Fit the histogram datapoints to an exponential
        #fit = scipy.optimize.curve_fit(fit_func, x, y, [-4.68*10**(-4), -2.15], error_of_mean_by_bin)
        fit = scipy.optimize.curve_fit(fit_func, x, y, [4.68*10**(-4)], error_of_mean_by_bin)
        #print('Exponential fit: y=Exp[%f * x + %f]' % (fit[0][0]*0.917, fit[0][1]*0.917))
        print('Exponential fit: y=Exp[%.9f * x + %f]' % (-fit[0][0], c_func(fit[0][0], sum(dxs)/len(dxs), x2)))

        print('L=%f' % ((fit[0][0] / magic_ice_const)**(-1)))
        
        plt.plot(x, fit_func(x, fit[0][0]), 'r')
    
        #line = lambda x: 0.259*0.917+0.363*10**(-3)*0.917*x
        #plt.plot(x,line(x), 'r.', linewidth=2.0, label='Expected total losses')
        #plt.axis([1, 10**6, 10**(-4), 10**4])
        plt.legend(loc=3)
        plt.xlabel('Track Distance (m)')
        plt.ylabel('Fractional Energy Loss')
        plt.title(label)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.savefig(plotdir + 'fractional_energy_loss_' + re.sub(' ', '_', label.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}')) + '.png')
        plt.clf()
        plt.close(fig)
    #plt.show()

def add_points(losses, weights, checkpoints, mu_info, points_functions, hists, E_bins, sample_d = 10, sample_E = 10):
    """
    Add points to the histogram for each muon with appropriate weighting
    """
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

        # Sanity check on the segment of track we are considering
        if max_range - min_range > 10000:
            print("Range greater than 10000m!")

        if max_range - min_range < max(E_bins):
            continue

        #if get_energy(min_range + max(E_bins), new_cps, loss_tuples, has_sum=True, inclusive=False) < 1000:
        #    continue

        # Keep track of losses not covered already
        next_losses = [loss for loss in sorted(loss_tuples, key=_get1) if loss[1] >= min_range and loss[1] <= max_range]
        
        # Convert to tuple for memoization
        loss_tuples = tuple(loss_tuples)

        for i in xrange(len(hists)):
            for x1,x2 in itertools.izip(hists[i].bins[:-1], hists[i].bins[1:]):
                x1 += min_range
                x2 += min_range
                if(x1 > max_range):
                    x = x1
                    dE = 0
                else:
                    if(x2 > max_range):
                        x2 = max_range
                    x, dE = points_functions[i]((x1, x2, new_cps, loss_tuples))
                #print(x, dE)
                point_x[i].append(x)
                point_dE[i].append(dE)
                point_weight[i].append(weight)
                #if(x2 == max_range):
                #    break
            total_dE = sum(point_dE[i])
            if(total_dE == 0):
                continue
            point_frac_dE[i] = [dE / total_dE for dE in point_dE[i]]
            point_x[i] = [x - min_range for x in point_x[i]]

            hists[i].add(point_x[i], point_frac_dE[i], point_weight[i])

        n_muons += 1

        if not n_muons % 1000:
            print("%d muons in file" % n_muons)

        # Clear the memoization dictionary for get_loss_info since we are done with the muon
        mlc.get_loss_info.__self__.clear()
        
    return n_muons

def get_hists_from_dirs(indirs, E_bins, points_functions):
    """
    Compute histograms from pickle contained the list of input directories
    """
    # Get list of input pickle files from all input directories
    infiles = [file for indir in indirs for file in glob.glob('%s*.pkl' % indir)]
    infiles.sort()

    n_muons = 0

    # Create a histogram object for each function
    hists = []
    for i in points_functions:
        hists.append(mlc.histogram(E_bins))

    # Loop over input files
    for infile in infiles[:6]:
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

points_functions = [get_total_losses_point, get_stoch_losses_point, get_mc_stoch_losses_point, get_cont_losses_point]
points_labels = ['MC total losses', 'MC stochastic losses', 'MC stochastic losses w/ ionization', 'MC continuous losses']

dist_bins = np.linspace(0, 600, 10+1)

if infile == '':
    if len(indirs):
        hists = get_hists_from_dirs(indirs, dist_bins, points_functions)
        if outfile != '':
            mlc.save_info_to_file(outfile, hists, dist_bins)
else:
    hists, dist_bins = mlc.get_info_from_file(infile)

plot(hists, dist_bins, points_labels, plotdir)

