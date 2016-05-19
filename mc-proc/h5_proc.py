import tables
import itertools
import numpy as np
import matplotlib.pyplot as plt
from icecube import dataclasses

file = tables.openFile('out.h5')

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

skip_muon = False
fractional_losses = []
distances = []
losses = []
init_muon_pos = np.arange(3)
is_first = True

for col in file.root.HighEMuonLosses.cols:
    if col[iv] == 0: #Is a muon
        init_muon_pos = np.array([col[ix], col[iy], col[iz]])
    else: #Is a loss
        if skip_muon:
            continue
        pos = np.array([col[ix], col[iy], col[iz]])
        if(col[iv] == 1 ):
            last_tot_loss = sum(losses)
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
        distance = np.linalg.norm(pos - init_muon_pos)
        loss = col[ie]
        if(distance < max_distance):
            distances.append(distance)
            losses.append(loss)

bin_it = np.linspace(0, max_distance, 31)
hists = [np.array([sum([loss for loss,dist in itertools.izip(frac_loss[0], frac_loss[1]) if dist >= bin and dist < bin+20]) for bin in bin_it]) for frac_loss in fractional_losses]
single_hist = sum(hists) / len(hists)
plt.hist(bin_it, bins=bin_it, weights=single_hist, color='b', histtype='step')
plt.title('Fractional muon energy loss along track')
plt.xlabel('Track distance (m)')
plt.ylabel('Average fractional energy loss')
plt.show()
