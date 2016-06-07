import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from icecube import dataclasses

import os
import pickle
import glob
import argparse
import itertools
import re

# Plots fractional muon energy loss as a fucntion of distance
# Takes fractional energy loss and position information from pickle files
# Pickle:
    # fractional_losses: [[(muon_0_fractional_loss_0, muon_0_track_distance_0), (m_0_fl_1, m_0_d_1), ...], [(m_1_fl_0, m_1_d_0), (m_1_fl_1, m_1_d_1), ...], ...]
    # weights: [muon_0_weight, muon_1_weight, ...]

# Set up argument parser
# Output directory stores nothing
# Max distance is the maximum distance considered for energy loss
# Bin size is the size of the distance bins in meters
parser = argparse.ArgumentParser(description='Proccess filenames')
parser.add_argument('indirs', metavar='indirs',  nargs='*')
parser.add_argument('-o', '--outfile', default='')
parser.add_argument('-i', '--infile', default='')
parser.add_argument('-d', '--max-distance', type=int, default=600)
parser.add_argument('-b', '--bin-size', type=int, default=60)
#parser.add_argument('-e,', '--exclude', default='deltaE')
parser.add_argument('-e,', '--exclude', default='')

args = parser.parse_args()
indirs = args.indirs
outfile = args.outfile
infile = args.infile
max_distance = args.max_distance
bin_size = args.bin_size
exclude = re.split('\W+', args.exclude)

# Create dictionary to get particle type numbers by name
particle_dict = {}
p = dataclasses.I3Particle()
for attr in dir(p.ParticleType):
    if not callable(attr) and not attr.startswith("__"):
        try:
            particle_dict[attr.lower()]=int(getattr(p.ParticleType, attr))
        except:
            pass
exclude = set([particle_dict[e.lower()] for e in exclude if e != ''])

def get_energy(x, checkpoints, losses):
    x = max(0, x)
    checkpoints = sorted(checkpoints, key=lambda cp: cp[1]) # Sort by distance
    losses = sorted(losses, key=lambda loss: loss[1]) # Sort by distance
    cp1 = [(e, d) for e,d in checkpoints if d <= x ][-1]
    greater_checkpoints = [(e, d) for e,d in checkpoints if d > x ]
    if x == cp1[1]:
        return cp1[0]
    elif len(greater_checkpoints) == 0:
        return 0
    cp2 = greater_checkpoints[0]
    losses = [l for l in losses if l[1] > cp1[1] and l[1] < cp2[1]]
    total_stochastic_loss = sum([l[0] for l in losses])
    loss_rate = (cp1[0] - cp2[0] - total_stochastic_loss) / (cp2[1] - cp1[1])

    losses = [l for l in losses if l[1] <= x]

    energy = cp1[0] - sum([l[0] for l in losses]) - (x - cp1[1]) * loss_rate

    return energy

def get_info_from_dirs(indirs, bin_it):
    # Get list of input pickle files from all input directories
    infiles = [file for indir in indirs for file in glob.glob('%s*.pkl' % indir)]
    infiles.sort()

    # Fractional energy loss plot
    weighted_file_hists = []
    weighted_file_sq_hists = []
    file_weights = []
    sum_weight_sq = 0

    n_muons = 0

    # Loop over input files
    for infile in infiles:
        if os.stat(infile).st_size == 0:
            continue
        print(infile)
        # Load fractional losses for each muon
        inpickle = open(infile, 'rb')
        losses = pickle.load(inpickle)
        n_muons += len(losses)

        # Remove losses beyond max distance
        losses = [[(loss, dist, type) for loss,dist,type in loss_tuples if dist < max_distance] for loss_tuples in losses]

        # Remove excluded loss types
        losses = [[(loss, dist) for loss,dist,type in loss_tuples if type not in exclude] for loss_tuples in losses]

        # Calculate total losses for each muon
        total_losses = np.array([sum([loss for loss,dist in loss_tuples]) for loss_tuples in losses])

        mask = np.array([total_loss > 0 for total_loss in total_losses])

        # Calculate histogram for each muon: each bin contains sum of fractional losses that occured in the bin's distance range
        hists = np.array([np.array([sum([loss for loss,dist in loss_tuples if dist >= bin and dist < bin+bin_size])/total_loss for bin in bin_it]) if total_loss > 0 else np.zeros(len(bin_it)) for loss_tuples,total_loss in itertools.izip(losses, total_losses)])

        # Load weights for each muon
        weights = np.array(pickle.load(inpickle))
        masked_weights = weights * mask

        # Load track energy checkpoints for each muon
        #checkpoints = np.array(pickle.load(inpickle))

        # Single histogram is the dot product of weights and hists
        # Together these 3  cover all the information needed from a file
        single_hist = np.dot(masked_weights,hists)
        single_sq_hist = np.dot(masked_weights, hists*hists)
        weight = sum(masked_weights)
        sum_weight_sq += np.dot(masked_weights,masked_weights)

        # Store the histogram and weight
        weighted_file_hists.append(single_hist)
        weighted_file_sq_hists.append(single_sq_hist)
        file_weights.append(weight)

        inpickle.close()

    # Final histogram is the sum of histograms from all files divided by total weight
    single_hist = sum(weighted_file_hists) / sum(file_weights)
    single_sq_hist = sum(weighted_file_sq_hists) / sum(file_weights)

    statistical_dev = [np.sqrt(sum_weight_sq) / sum(file_weights)] * (len(single_hist) - 1)
    std_dev = np.sqrt(single_sq_hist - single_hist**2)

    info = (single_hist, statistical_dev, std_dev, n_muons)
    return info

def get_info_from_file(infile):
    inpickle = open(infile, 'rb')
    info = pickle.load(inpickle)
    bin_it = pickle.load(inpickle)
    inpickle.close()
    return (info, bin_it)

def save_info_to_file(outfile, info, bin_it):
    outpickle = open(outfile, 'wb')
    pickle.dump(info, outpickle, -1)
    pickle.dump(bin_it, outpickle, -1)
    outpickle.close()

def fit_func(x, b, a):
    return np.exp(x*b+a)

# Create the distance bins
bin_it = np.linspace(0, max_distance, max_distance / bin_size + 1)

if infile == '':
    if len(indirs):
        info = get_info_from_dirs(indirs, bin_it)
        if outfile != '':
            save_info_to_file(outfile, info, bin_it)
else:
    info, bin_it = get_info_from_file(infile)

single_hist, statistical_dev, std_dev, n_muons = info

# Fit the histogram datapoints to an exponential
fit = scipy.optimize.curve_fit(fit_func, ((bin_it[:-1] + bin_it[1:]) / 2), single_hist[:-1], [-4.68*10**(-4), -2.15], statistical_dev)
print('Exponential fit: y=Exp[%f * x + %f]' % (fit[0][0], fit[0][1]))
print('%d muons' % n_muons)

print('Error: sigma=%f' % statistical_dev[0])
print('L=%f' % (-fit[0][0]**(-1)))

# Plot histogram with error bars
#plt.errorbar(((bin_it[:-1] + bin_it[1:]) / 2), single_hist[:-1], yerr=std_dev[:-1], fmt="_")
plt.hist(bin_it, bins=bin_it, weights=single_hist, color='b', histtype='step')
plt.title('Fractional muon energy loss along track')
plt.xlabel('Track distance (m)')
plt.ylabel('Average fractional energy loss')
plt.show()
