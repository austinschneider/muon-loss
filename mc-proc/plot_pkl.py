import numpy as np
import matplotlib.pyplot as plt

import pickle
import glob
import argparse
import itertools

parser = argparse.ArgumentParser(description='Proccess filenames')

parser.add_argument('indirs', metavar='indirs',  nargs='+')

args = parser.parse_args()
indirs = args.indirs

infiles = [file  for indir in indirs for file in glob.glob('%s*.pkl' % indir)]
infiles.sort()

fractional_losses = []
weights = []

for infile in infiles:
    inpickle = open(infile, 'rb')
    fractional_losses += pickle.load(inpickle)
    weights += pickle.load(inpickle)
    inpickle.close()

max_distance = 600.

bin_size = 60

bin_it = np.linspace(0, max_distance, max_distance / bin_size + 1)
hists = [np.array([sum([loss for loss,dist in frac_loss if dist >= bin and dist < bin+bin_size]) for bin in bin_it]) for frac_loss in fractional_losses]
weighted_hists = [hist*weight for hist,weight in itertools.izip(hists, weights)]
single_hist = sum(weighted_hists) / sum(weights)
plt.hist(bin_it, bins=bin_it, weights=single_hist, color='b', histtype='step')
plt.title('Fractional muon energy loss along track')
plt.xlabel('Track distance (m)')
plt.ylabel('Average fractional energy loss')
plt.show()
