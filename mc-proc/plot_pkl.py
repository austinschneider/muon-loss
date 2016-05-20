import numpy as np
import matplotlib.pyplot as plt

import pickle
import glob
import argparse  

parser = argparse.ArgumentParser(description='Proccess filenames')

parser.add_argument('indirs', metavar='indirs',  nargs='+')
#parser.add_argument('-g', '--geo', default='')
#parser.add_argument('-o', '--outfile', default='')

args = parser.parse_args()
indirs = args.indirs

infiles = [file  for indir in indirs for file in glob.glob('%s*.pkl' % indir)]

fractional_losses = []

for infile in infiles:
    inpickle = open(infile, 'rb')
    fractional_losses += pickle.load(inpickle)
    inpickle.close()

max_distance = 600.

bin_it = np.linspace(0, max_distance, 31)
#hists = [np.array([sum([loss for loss,dist in itertools.izip(frac_loss[0], frac_loss[1]) if dist >= bin and dist < bin+20]) for bin in bin_it]) for frac_loss in fractional_losses]
hists = [np.array([sum([loss for loss,dist in frac_loss if dist >= bin and dist < bin+20]) for bin in bin_it]) for frac_loss in fractional_losses]
single_hist = sum(hists) / len(hists)
plt.hist(bin_it, bins=bin_it, weights=single_hist, color='b', histtype='step')
plt.title('Fractional muon energy loss along track')
plt.xlabel('Track distance (m)')
plt.ylabel('Average fractional energy loss')
plt.show()
