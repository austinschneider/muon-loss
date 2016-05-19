import tables
import itertools
import numpy as np
from icecube import dataclasses

file = tables.openFile('out.h5')

max_distance = 600.
min_energy = 5000.

coldict = {}
colnames = file.root.HighEMuonLosses.colnames
for i in range(0, len(colnames)):
    coldict[colnames[i]] = i

iv = coldict['vector_index']
ix = coldict['x']
iy = coldict['y']
iz = coldict['z']
ie = coldict['energy']

skip_muon = False
fractional_losses = []
distances = []
losses = []
init_loss_pos = np.arange(3)
is_first = True

for col in file.root.HighEMuonLosses.cols:
    if col[iv] == 0: #Is a muon
        skip_muon = col[ie] < min_energy # 5TeV cut
        last_tot_loss = sum(losses)
    else: #Is a loss
        if skip_muon:
            continue
        pos = np.array([col[ix], col[iy], col[iz]])
        if(col[iv] == 1 ):
            if last_tot_loss > 0:
                last_frac_losses = np.array(losses) / last_tot_loss
                fractional_losses.append((last_frac_losses, distances))
            else:
                if is_first:
                    is_first = False
                else:
                    print "Encountered lossless muon!"
            distances = []
            losses = []
            init_loss_pos = pos
        distance = np.linalg.norm(pos - init_loss_pos)
        loss = col[ie]
        if(distance <= max_distance):
            distances.append(distance)
            losses.append(loss)

bin_it = np.linspace(0, 600, 600./20.)
hists = [np.array([sum([loss for loss,dist in itertools.izip(frac_loss[0], frac_loss[1]) if dist > bin and dist <= bin+20]) for bin in bin_it]) for frac_loss in fractional_losses]
single_hist = sum(hists) / len(hists)
counts, bins, patches = pylab.hist(single_hist, bin_it, color='r', histtype='step')
