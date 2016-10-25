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

import frac_loss
import energy_distribution

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

def plot(hists, binnings, plot_functions, plotdir):
    """
    Call plotting functions on all histograms
    """
    for func, hist, bins in itertools.izip(plot_functions, hists, binnings):
        func(hist, bins, plotdir)

def get_hists_from_files(infiles, binnings, points_functions, hists, file_range, n = 1):
    """
    Process information from input files to create histograms
    """
    reader_hists = [hists] + [[copy.deepcopy(hist) for hist in hists] for i in xrange(n-1)]
    # Loop over input files
    for f in infiles:
        m = multiprocessing.Manager()
        q = m.Queue(maxsize=n)
        reader_pool = multiprocessing.Pool(1)
        reader_result = reader_pool.apply_async(mlc.file_reader, (f, q))
        #pool = multiprocessing.Pool(n)
        #threads = [multiprocessing.Process(target=get_hists_from_reader, args=(r,binnings,points_functions,h)) for r,h in itertools.izip(readers, reader_hists)]
        #results = [pool.apply_async(get_hists_from_queue, (q,binnings,points_functions,h,i)) for i,h in enumerate(reader_hists)]
        reader_hists = [get_hists_from_queue(q, binnings, points_functions, h, i)]
        reader_result.get()                                                                                                                                                                    
        #reader_hists = [res.get() for res in results]
    for h in reader_hists[1:]:
        for i in xrange(len(h)):
            reader_hists[0][i].accumulate(h[i])

    return reader_hists[0]


def get_hists_from_queue(q, binnings, points_functions, hists, q_n):
    print 'In thread!'
    elems = []
    done = False
    n_good = 0
    n_bad = 0
    try:
        while True:
            if len(elems) == 0:
                try:
                    elems_string = q.get(True, 1)
                    if type(elems_string) is list:
                        elems = elems_string
                    elif type(elems_string) == str:
                        if elems_string == '':
                            print 'Ending get_hists_from_queue'
                            q.put('')
                            print 'Put Empty'
                            break
                        try:
                            elems = json.loads(elems_string)
                        except:
                            continue
                    if len(elems) == 0:
                        continue
                    done = False
                except Queue.Empty as e:
                    if done == False:
                        done = True
                    continue
            data = elems.pop()
            
            loss_tuples, weight, cps, mu, nu, run_id, event_id = data # Unpack the data
            loss_tuples = mlc.add_loss_sum(loss_tuples, mlc.get_valid_checkpoints(cps)) # Pre-calculate loss sums

            # Make everything a tuple for memoization
            loss_tuples = tuple([tuple(loss) for loss in loss_tuples])
            cps = tuple([tuple(cp) for cp in cps])
            mu = tuple(mu)
            nu = tuple(nu)

            # Add points to each histogram
            for hist, func, bins in itertools.izip(hists, points_functions, binnings):
                ret = func(hist, bins, loss_tuples, weight, cps, mu, nu, run_id, event_id)
                if ret:
                    n_good += 1
                else:
                    n_bad += 1

            # Clear the memoization dictionaries since we are done with the muon
            mlc.get_loss_info.__self__.clear()
            mlc.get_energy_.__self__.clear()
    except Exception as e:
        print 'Got exception in thread %d' % q_n
        print e
    print 'Good: %d' % n_good
    print 'Bad: %d' % n_bad
    return hists

# Create the binnings we want to work with
bins_600 = np.linspace(0, 600, 10+1)
bins_1560 = np.linspace(0, 1560, 26+1)
half_bins_600 = np.linspace(0, 600, 2+1)
half_bins_1560 = np.linspace(0, 1560, 2+1)

# Specify the functions we will use to add points to histograms
points_functions = [frac_loss.add_point, frac_loss.add_point, frac_loss.add_point, frac_loss.add_point] + [energy_distribution.add_point, energy_distribution.add_point]

# Specify the binnings
binnings = [bins_600, bins_1560, half_bins_600, half_bins_1560] + [bins_600, bins_1560]

# Create the histogram objects to store information
hists = [frac_loss.make_hist(bins) for bins in binnings[:-2]] + [energy_distribution.make_hist(bins) for bins in binnings[-2:]]

# Specify the plotting functions
plot_functions = [frac_loss.plot]*4 + [energy_distribution.plot]*2

# Do different things depending on the arguments
if plotdir != '' or outfile != '':
    if len(hist_file) == 0 and len(infiles) > 0:
        if aggregate:
            hists, binnings = mlc.aggregate_info_from_files(infiles)
        else:
            hists = get_hists_from_files(infiles, binnings, points_functions, hists, file_range, n=24)
        if outfile != '':
            mlc.save_info_to_file(outfile, hists, binnings)
    elif len(hist_file) > 0:
        hists, binnings = mlc.get_info_from_file(hist_file)
    if outfile != '':
        mlc.save_info_to_file(outfile, hists, binnings)
    if plotdir != '':
        plot(hists, binnings, plot_functions, plotdir)
