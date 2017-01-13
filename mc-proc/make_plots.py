#!/usr/bin/env python
import numpy as np

from icecube import dataclasses

import copy
import glob
import argparse
import itertools
import threading
import multiprocessing
import Queue

import muon_loss_common as mlc
from icecube import NewNuFlux
import icecube.weighting.weighting as weighting
import frac_loss
import energy_distribution
import cut_distribution

import json


# Set up the argument parser
parser = argparse.ArgumentParser(description='Proccess filenames')
parser.add_argument('-i', '--infiles', metavar='infiles', type=str, default='')
parser.add_argument('-o', '--outfile', default='')
parser.add_argument('-f', '--histogram-file', dest='hist_file', default='')
parser.add_argument('-p', '--plotdir', default='')

parser.add_argument('--aggregate', dest='aggregate', action='store_true')
parser.add_argument('--no-aggregate', dest='aggregate', action='store_false')
parser.set_defaults(aggregate=False)

parser.add_argument('-r', '--range', default='')
parser.add_argument('-n', '--ncores', type=int, default=1)
parser.add_argument('-s', '--sample', type=float, default=1.0)

args = parser.parse_args()
infile_string = args.infiles.strip('"\'')
infiles = glob.glob(infile_string)
outfile = args.outfile.strip('"\'')
print args
hist_file = args.hist_file.strip('"\'')
plotdir = args.plotdir.strip('"\'')
aggregate = args.aggregate
file_range = args.range.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}_ ') # Remove anything weird before storing
if file_range == '':
    file_range = None
else:
    file_range = [int(i) for i in range.split('-', 1)] # Split into 1 or 2 numbers
ncores = args.ncores
sampling_factor = (lambda x: min(x, 1.0/x))(args.sample)

#Define the flux and the generator

flux_name = 'honda2006'                                                                                                                     
generator = weighting.NeutrinoGenerator(1000000, 1000, 1000000, -2, 'NuMu', InjectionMode='Surface')

def plot(hists, binnings, plot_functions, plotdir):
    """
    Call plotting functions on all histograms
    """
    for func, hist, bins in itertools.izip(plot_functions, hists, binnings):
        func(hist, bins, plotdir)

# Create the binnings we want to work with
bins_600 = np.linspace(0, 600, 10+1)
bins_1560 = np.linspace(0, 1560, 26+1)
half_bins_600 = np.linspace(0, 600, 2+1)
half_bins_1560 = np.linspace(0, 1560, 2+1)

# Specify the functions we will use to add points to histograms
points_functions = [frac_loss.add_point, frac_loss.add_point, frac_loss.add_point, frac_loss.add_point] + [energy_distribution.add_point, energy_distribution.add_point] + [cut_distribution.add_point, cut_distribution.add_point]

# Specify the binnings
binnings = [bins_600, bins_1560, half_bins_600, half_bins_1560] + [bins_600, bins_1560] + [bins_600, bins_1560]

# Create the histogram objects to store information
hists = [frac_loss.make_hist(bins) for bins in binnings[:-4]] + [energy_distribution.make_hist(bins) for bins in binnings[-4:-2]] + [cut_distribution.make_hist(bins) for bins in binnings[-2:]]

# Specify the plotting functions
plot_functions = [frac_loss.plot]*4 + [energy_distribution.plot]*2 + [cut_distribution.plot]*2

# Do different things depending on the arguments
if plotdir != '' or outfile != '':
    if len(hist_file) == 0 and len(infiles) > 0:
        if aggregate:
            hists, binnings = mlc.aggregate_info_from_files(infiles)
        else:
            hists = mlc.get_hists_from_files(infiles, binnings, points_functions, hists, file_range, ncores, flux_name, generator, sampling_factor)
        if outfile != '':
            mlc.save_info_to_file(outfile, hists, binnings)
    elif len(hist_file) > 0:
        hists, binnings = mlc.get_info_from_file(hist_file)
    if outfile != '':
        mlc.save_info_to_file(outfile, hists, binnings)
    if plotdir != '':
        plot(hists, binnings, plot_functions, plotdir)
