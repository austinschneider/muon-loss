import tables
import itertools
import numpy as np
import matplotlib.pyplot as plt

from icecube import dataclasses
from icecube import NewNuFlux
from icecube.icetray import I3Units
from icecube.weighting.weighting import from_simprod

import argparse
import pickle
import ntpath
import glob

parser = argparse.ArgumentParser(description='Proccess filenames')

parser.add_argument('indirs', metavar='indirs',  nargs='+')
#parser.add_argument('-i', '--infile', default='')
parser.add_argument('-o', '--outdir', default='./pickle')

args = parser.parse_args()
indirs = args.indirs
outdir = args.outdir

max_distance = 600.

flux = NewNuFlux.makeFlux('honda2006').getFlux

generator = ""

# Get a set of output files
def get_pickled_files(dir):
    processed_files = set()
    pickles = glob.glob('%s*.pkl' % dir)
    for p in pickles:
        pickle_source = ntpath.splitext(ntpath.basename(p))[0]
        processed_files.add(pickle_source)
    return processed_files

def get_fractional_losses(infile):
    try:
        file = tables.openFile(infile)
    except:
        return []

    max_distance = 600.

    coldict = {}
    colnames = file.root.HighEMuonLosses.colnames
    for i in range(0, len(colnames)):
        coldict[colnames[i]] = i

    iv = coldict['vector_index']
    ix = coldict['x']
    iy = coldict['y']
    iz = coldict['z']
    ie = coldict['energy']
    it = coldict['type']
    izen = coldict['zenith']
    ievent = coldict['Event']

    fractional_losses = []
    weights = []
    losses = []
    distances= []
    init_muon_pos = np.arange(3)
    is_first = True

    n_nu = -1
    weight = 0

    for col in file.root.HighEMuonLosses.cols:
        if col[iv] == 0: #Is a neutrino
            n_nu += 1
            if(col[ievent] != file.root.I3MCWeightDict.cols.Event[n_nu]):
                raise ValueError('Event numbers do not match: (%d,%d) %d' % (col[ievent], file.root.I3MCWeightDict.cols.Event[n_nu], n_nu))
            generator = from_simprod(11374)
            energy = col[ie]
            type = col[it]
            zenith = col[izen]
            cos_theta = np.cos(zenith)
            p_int = file.root.I3MCWeightDict.cols.TotalInteractionProbabilityWeight[n_nu]
            unit = I3Units.cm2/I3Units.m2
            weight = p_int*(flux(type, energy, cos_theta)/unit)/generator(energy, type, cos_theta)
        elif col[iv] == 1: #Is a muon
            init_muon_pos = np.array([col[ix], col[iy], col[iz]])
        else: #Is a loss
            pos = np.array([col[ix], col[iy], col[iz]])
            if(col[iv] == 2 ):
                total_loss = sum(losses)
                if total_loss > 0:
                    frac_losses = [loss / total_loss for loss in losses]
                    fractional_losses.append([(loss, dist) for loss,dist in itertools.izip(frac_losses, distances)])
                    weights.append(weight)
                else:
                    if is_first:
                        is_first = False
                    else:
                        print "Encountered lossless muon!"
                distances = []
                losses = []
            distance = np.linalg.norm(pos - init_muon_pos)
            loss = col[ie]
            if(distance < max_distance):
                losses.append(loss)
                distances.append(distance)

    total_loss = sum(losses)
    if total_loss > 0:
        frac_losses = [loss / total_loss for loss in losses]
        fractional_losses.append([(loss, dist)for loss,dist in itertools.izip(frac_losses, distances)])
        weights.append(weight)
    else:
        if is_first:
            is_first = False
        else:
            print "Encountered lossless muon!"

    #print([sum([loss for loss,dist in frac_loss]) for frac_loss in fractional_losses if sum([loss for loss,dist in frac_loss]) > 1.00001])
               
    file.close()

    outpickle = open('%s%s.pkl' % (outdir, ntpath.basename(infile)), 'wb')
    pickle.dump(fractional_losses, outpickle, -1)
    pickle.dump(weights, outpickle, -1)
    outpickle.close()

    #print(max([(i,losses) for i,losses in itertools.izip(range(0,len(fractional_losses)), fractional_losses)], key=(lambda x: max(x[1], key=(lambda y: y[0])))))

    return fractional_losses

processed_files = get_pickled_files(outdir)
infiles = [file for indir in indirs for file in glob.glob('%s*.h5' % indir) if ntpath.basename(file) not in processed_files]
infiles.sort()

for infile in infiles:
    try:
        frac_loss = get_fractional_losses(infile)
    except:
        frac_loss = []
    print 'Got %d more muons!' % len(frac_loss)

#print 'Have %d muons' % len(fractional_losses)

#bin_it = np.linspace(0, max_distance, 31)
##hists = [np.array([sum([loss for loss,dist in itertools.izip(frac_loss[0], frac_loss[1]) if dist >= bin and dist < bin+20]) for bin in bin_it]) for frac_loss in fractional_losses]
#hists = [np.array([sum([loss for loss,dist in frac_loss if dist >= bin and dist < bin+20]) for bin in bin_it]) for frac_loss in fractional_losses]
#single_hist = sum(hists) / len(hists)
#plt.hist(bin_it, bins=bin_it, weights=single_hist, color='b', histtype='step')
#plt.title('Fractional muon energy loss along track')
#plt.xlabel('Track distance (m)')
#plt.ylabel('Average fractional energy loss')
#plt.show()
