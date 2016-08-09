#!/usr/bin/env python
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
histogram = mlc.histogram

# Set up argument parser
# Input directories where the pickles are found
# Outfile to store histogram information
# Plot directory to store plots in
parser = argparse.ArgumentParser(description='Proccess filenames')
parser.add_argument('indirs', metavar='indirs',  nargs='*')
parser.add_argument('-o', '--outfile', default='')
parser.add_argument('-i', '--infile', default='')
parser.add_argument('-p', '--plotdir', default='')

parser.add_argument('--aggregate', dest='aggregate', action='store_true')
parser.add_argument('--no-aggregate', dest='aggregate', action='store_false')
parser.set_defaults(aggregate=True)

parser.add_argument('-r', '--range', default='')
parser.add_argument('-d', '--decadefraction', type=int, default=2)

# Parse the arguments
args = parser.parse_args()
#args = parser.parse_args(['/data/user/aschneider/muon_energy_loss/combo_trunk/5cp_E_cut/', '-p', './plots_5cp_E_cut_memtest/'])
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
decade_fraction = int(args.decadefraction)

# Define fucntion to get item from list or tuple
_get0 = op.itemgetter(0)
_get1 = op.itemgetter(1)

def plot_dEdx(hists, E_bins, points_labels, plotdir, split=False):

    colors = ['b', 'g', 'm', 'c']
    
    #fig = plt.figure()
    fig, ax = plt.subplots(1)

    a = 0.259
    b = 0.363*10**(-3)
    line = lambda x: a*0.917+b*0.917*x
    x = np.logspace(np.log10(min(E_bins)), np.log10(max(E_bins)), (lambda l: min(l[1:] - l[:-1]) * 100 * (len(l) - 1))(np.log10(np.array(sorted(E_bins))))+1)
    plt.plot(x, line(x), 'r', linewidth=1.0, label='Expected total losses')

    L = None

    for hist,label,color in itertools.izip(hists, points_labels, colors):
        print(label)
        
        x = hist.get_x()
        dEdx, dE, dx, Ex = hist.get_y()

        dEdx_e, dE_e, dx_e, Ex_e = hist.get_y_stddev_of_mean()
        
        Ex[dx == 0] = 0
        dx_e[dx == 0] = 0
        dE_e[dE == 0] = 0
        dx[dx == 0] = 1

        if split:
            dEdx = dE / dx
            dE[dE == 0] = 1
            dEdx_e = abs(dEdx) * np.sqrt((dE_e/abs(dE))**2 + (dx_e/abs(dx))**2)

        x = Ex / dx

        print x <= E_bins[1:]
        print x >= E_bins[:-1]

        mask = hist.bins[:-1] >= 1000

        magic_ice_const = 0.917

        fit_func = lambda x, b, a: x*b*magic_ice_const+a*magic_ice_const
        fit = scipy.optimize.curve_fit(fit_func, x[mask], dEdx[mask], [4.68*10**(-4), -2.15], dEdx_e[mask])
        print('Fit: y=%f * x + %f' % (fit[0][0]*0.917, fit[0][1]*0.917))

        if L is None:
            L = (fit[0][0])**(-1)

        print('L=%f' % ((fit[0][0])**(-1)))

        neg = False
        for i in xrange(len(dEdx)):
            if dEdx[i] < 0:
                neg = True
        if neg:
            continue
   
        plt.errorbar(x, dEdx, fmt=color+'.', label=label, yerr=[dEdx_e, dEdx_e])
        try:
            plt.hist(x, bins=E_bins, weights=dEdx, color=color, histtype='step')
        except:
            print(label)
            print("x: ", x)
            print("y: ", dEdx)
            raise

    #fig, ax = plt.subplots(1)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.70, 0.1, "L=%.03f (m)" % L, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
   
    plt.axis([1, 10**6, 10**(-4), 10**4])
    plt.legend(loc=2)
    plt.xlabel('Muon Energy (GeV)')
    plt.ylabel('dE/dx (GeV/m)')
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    if split:
        plt.savefig(plotdir + 'dEdx_%fGeV-%fGeV_%dbins_split.png' % (E_bins[0], E_bins[-1], len(E_bins) - 1) )
    else:
        plt.savefig(plotdir + 'dEdx_%fGeV-%fGeV_%dbins.png' % (E_bins[0], E_bins[-1], len(E_bins) - 1) )
    plt.close(fig)

def add_E_dEdx_points(losses, weights, checkpoints, mu_info, points_functions, hists, E_bins, sample_d = 10, sample_E = 10):
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
        point_E = np.zeros((len(hists), 0)).tolist()
        point_dEdx = np.zeros((len(hists), 0)).tolist()
        point_dE = np.zeros((len(hists), 0)).tolist()
        point_dx = np.zeros((len(hists), 0)).tolist()
        point_Ex = np.zeros((len(hists), 0)).tolist()
        point_weight = np.zeros((len(hists), 0)).tolist()
        point_bin = np.zeros((len(hists), 0)).tolist()

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

        # Keep track of losses not covered already
        next_losses = [loss for loss in sorted(list(loss_tuples), key=_get1) if loss[1] >= min_range and loss[1] <= max_range]
        
        # Convert to tuple for memoization
        loss_tuples = tuple(loss_tuples)


        # Try to get a (E, dEdx) point for each energy bin
        x1 = min_range
        x2 = x1
        while(x2 < max_range):
            try:
                # Get energy at beginning of bin
                E1 = mlc.get_energy(x1, new_cps, loss_tuples, inclusive=True)

                # Energy that goes to the next bin
                Ei, E2 = next((E for E in reversed(list(enumerate(E_bins))) if E[1] < E1), (None, None))

                if not Ei:
                    break

                if np.isclose(E1, E2, rtol=1e-05, atol=1e-09):
                    Ei, E2 = next((E for E in reversed(list(enumerate(E_bins))) if E[1] < E2), (None, None))

                if not Ei:
                    break



                x2 = None

                # Loop over losses infront of x1 that are within the bounds
                for loss in next_losses:
                    # Check to see if the loss takes us to the next bin
                    E_after_loss = mlc.get_energy(loss[1], new_cps, loss_tuples, inclusive=True)
                    if E_after_loss < E2:
                        E_before_loss = mlc.get_energy(loss[1], new_cps, loss_tuples, inclusive=False)
                        x2 = loss[1]

                        # If the energy we want is in the region before the loss, find x2 using the loss rate
                        # Otherwise x2 is at the loss position
                        if E_before_loss < E2:
                            # Find the position x2 that corresponds to energy E2
                            # Get the loss rate at E2
                            cp2, cp1 = mlc.get_bounding_elements(E2, new_cps, key=_get0, sort=True)
                            if cp1 == cp2:
                                x2 = cp2[1]
                            else:
                                loss_rate, losses = mlc.get_loss_info((cp1, cp2, loss_tuples, True))

                                # Get valid (energy, distance) in front of E2 to extrapolate from
                                cp2, cp1 = mlc.get_bounding_elements(E2, [(E_before_loss, loss[1], "loss")] + list(new_cps), key=_get0, sort=True)

                                E0 = cp2[0]
                                x0 = cp2[1]

                                x2 = (E2 - E0) / (-loss_rate) + x0
                            
                            E_x2 = mlc.get_energy(x2, new_cps, loss_tuples, inclusive=True)

                            # Make sure that the energy at x2 is the same as E2
                            if not np.isclose(E_x2, E2, rtol=1e-05, atol=1e-09):
                                print("Not close to E2!")
                                print("E2: %f" % E2, "E_x2: %f" % E_x2, "E0: %f" % E0)
                                print("x2: %f" % x2, "x0: %f" % x0)
                                print("Bounding: ", cp1, cp2)
                                print("loss_rate: %f" % loss_rate)
                                raise ValueError("E_x2 is not close to E2!")
                            
                        # Break out of the loop because we determined x2
                        break

                if x2 is None:
                    cp2, cp1 = mlc.get_bounding_elements(E2, new_cps, key=_get0, sort=True)
                    if cp2 is None:
                        x2 = max_range
                    else:
                        #cp2, cp1 = mlc.get_bounding_elements(E2, new_cps, key=_get0, sort=True)
                        if cp1 == cp2:
                            x2 = cp2[1]
                        else:
                            loss_rate, losses = mlc.get_loss_info((cp1, cp2, loss_tuples, True))

                            # Get valid (energy, distance) in front of E2 to extrapolate from
                            cp2, cp1 = mlc.get_bounding_elements(E2, list(new_cps), key=_get0, sort=True)

                            E0 = cp2[0]
                            x0 = cp2[1]
                            
                            x2 = (E2 - E0) / (-loss_rate) + x0

                if x2 > max_range:
                    x2 = max_range
                
                # Make sure that the interval we found is meaningful
                if x1 == x2:
                    raise ValueError("Begin and end are equal: %f" % x1)
                elif np.isclose(x1, x2, rtol=1e-09, atol=1e-06):
                    raise ValueError("Begin and end are close: %f, %f" % (x1, x2))

                if E1 <= max(E_bins) and E1 >= min(E_bins):
                    Ex = mlc.integrate_track_energy(x1, x2, new_cps, loss_tuples)
                    # Get the dEdx point using predefined functions
                    # Input is single tuple to allow fast memoization if desired
                    for i in xrange(len(hists)):
                        E, dE, dx = points_functions[i]((x1, x2, new_cps, loss_tuples))
                        dEdx = dE / dx
                    
                        if Ex / dx > E1 or Ex / dx < E2:
                            print "Ex: %f" % Ex
                            print "dx: %f" % dx
                            print "Ex/dx: %f" % (Ex / dx)
                            raise ValueError("Average energy out of bounds")

                        # Accumulate list of points to add to histogram
                        point_E[i].append(E)
                        point_dEdx[i].append(dEdx)
                        point_dE[i].append(dE)
                        point_dx[i].append(dx)
                        point_Ex[i].append(Ex)
                        point_weight[i].append(weight)
                        point_bin[i].append(Ei)
                
            except:
                print("####")
                print("Checkpoints:")
                print(cps)
                print("New checkpoints:")
                print(new_cps)
                print("Loss tuples:")
                print(loss_tuples)
                print("Mu info:")
                print(mu)
                print("E1: %f" % E1, "E2: %f" % E2)
                print("x1: ", x1, "x2: ", x2)
                print("Next losses:")
                print(next_losses)
                raise

            # Maintain next_losses as losses that are in the current bin or further ahead
            next_losses = [loss for loss in next_losses if loss[1] > x2]

            x1 = x2

        n_muons += 1
        for i in xrange(len(hists)):
            hists[i].add(point_E[i], [point_dEdx[i], point_dE[i], point_dx[i], point_Ex[i]], point_weight[i], point_bin[i])

        # Clear the memoization dictionary for get_loss_info since we are done with the muon
        mlc.get_loss_info.__self__.clear()

        if not n_muons % 1000:
            print("%d muons in file" % n_muons)
        
    return n_muons

def get_hists_from_dirs(indirs, E_bins, points_functions, range=None):
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
        hists.append(mlc.histogram_nd(4, E_bins, store_data=False))

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
        n_muons += add_E_dEdx_points(losses, weights, checkpoints, mu_info, points_functions, hists, E_bins)

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
    return (E_at_x1, abs((E_at_x1 - E_at_x2)), (x2 - x1))

def get_stoch_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    return (mlc.get_energy(x1, cps, loss_tuples), abs(sum([loss[0] for loss in loss_tuples if loss[2] not in exclude and loss[1] > x1 and loss[1] <= x2])), (x2 - x1))

def get_mc_stoch_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    return (mlc.get_energy(x1, cps, loss_tuples), abs(sum([loss[0] for loss in loss_tuples if loss[1] > x1 and loss[1] <= x2])), (x2 - x1))

def get_cont_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    E_at_x1 = mlc.get_energy(x1, cps, loss_tuples)
    E_at_x2 = mlc.get_energy(x2, cps, loss_tuples)
    return (E_at_x1, (abs((E_at_x1 - E_at_x2)) - abs(sum([loss[0] for loss in loss_tuples if loss[1] > x1 and loss[1] <= x2]))), (x2 - x1))

points_functions = [get_total_losses_point, get_stoch_losses_point, get_mc_stoch_losses_point, get_cont_losses_point]
points_labels = ['MC total losses', 'MC stochastic losses', 'MC stochastic losses w/ ionization', 'MC continuous losses']


# Create log energy bins
min_exp = 0
max_exp = 6

E_bins = np.logspace(min_exp, max_exp, (max_exp - min_exp)*decade_fraction+1)

if plotdir != '' or outfile != '':
    if len(infile) == 0 and len(indirs) > 0:
        if aggregate:
            hists, E_bins = mlc.aggregate_info_from_dirs(indirs)
        else:
            hists = get_hists_from_dirs(indirs, E_bins, points_functions, range)
        if outfile != '':
            mlc.save_info_to_file(outfile, hists, E_bins)
    elif len(infile) > 0:
        hists, E_bins = mlc.get_info_from_file(infile)
    if outfile != '':
        mlc.save_info_to_file(outfile, hists, E_bins)
    if plotdir != '':
        plot_dEdx(hists, E_bins, points_labels, plotdir, split=True)

