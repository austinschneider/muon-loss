import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize

from icecube import dataclasses

import os
import pickle
import glob
import argparse
import itertools
import re
import operator as op

def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__


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

# Set up exclusion set to differentiate losses
exclude = ['deltaE']
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

sim_vol_length = 1600
sim_vol_radius = 800

sim_vol_top = sim_vol_length / 2
sim_vol_bottom = -sim_vol_top


muon_p_energy = 0
muon_p_pos_x = 1
muon_p_pos_y = 2
muon_p_pos_z = 3
muon_p_dir_zenith = 4
muon_p_dir_azimuth = 5
muon_p_length = 6

is_in_sim_vol = lambda m: (m[muon_p_pos_z] < sim_vol_top and m[muon_p_pos_z] > sim_vol_bottom and (m[muon_p_pos_x]**2 + m[muon_p_pos_y]**2)**(0.5) < sim_vol_radius)

class histogram:
    """ Histogram class for creating weighted average plot """
    def __init__(self, bins):
        self.bins = np.array(bins)
        size = len(bins) - 1
        self.weights = np.zeros(size)
        self.weights2 = np.zeros(size)
        self.x_weighted = np.zeros(size)
        self.y_weighted = np.zeros(size)
        self.x2_weighted = np.zeros(size)
        self.y2_weighted = np.zeros(size)
    def add(self, x, y, weights):
        x = np.array(x)
        y = np.array(y)
        weights = np.array(weights)

        weights_by_bin, edges = np.histogram(x, bins=self.bins, weights=weights)
        weights2_by_bin, edges = np.histogram(x, bins=self.bins, weights=weights**2)
        x_weighted_by_bin, edges = np.histogram(x, bins=self.bins, weights=x * weights)
        y_weighted_by_bin, edges = np.histogram(x, bins=self.bins, weights=y * weights)
        x2_weighted_by_bin, edges = np.histogram(x, bins=self.bins, weights=x**2 * weights)
        y2_weighted_by_bin, edges = np.histogram(x, bins=self.bins, weights=y**2 * weights)

        self.weights += weights_by_bin
        self.weights2 += weights2_by_bin
        self.x_weighted += x_weighted_by_bin
        self.y_weighted += y_weighted_by_bin
        self.x2_weighted += x2_weighted_by_bin
        self.y2_weighted += y2_weighted_by_bin

# Define fucntion to get [1] item from list or tuple
_get1 = op.itemgetter(1)

@memodict
def get_loss_info(stuff):
    """
    Get the loss rate and losses between two energy checkpoints.
    Takes single tuple argument (checkpoint1, checkpoint2, losses) to allow fast memoization.
    A checkpoint has the structure: (energy, distance along track)
    """
    cp1, cp2, losses = stuff
    losses = sorted(losses, key=_get1) # Sort by distance 
    losses = [l for l in losses if l[1] > cp1[1] and l[1] < cp2[1]]
    total_stochastic_loss = sum([l[0] for l in losses])
    loss_rate = (cp1[0] - cp2[0] - total_stochastic_loss) / (cp2[1] - cp1[1])

    return (loss_rate, losses)

def get_energy(x, checkpoints, losses):
    """
    Get the energy of a muon track at a point x.
    Given energy checkpoints and losses along track.
    """
    x = max(0, x) # Return zero for negative positions
    checkpoints = sorted(checkpoints, key=_get1) # Sort by distance
    cp1 = [(e, d) for e,d in checkpoints if d <= x ][-1] # checkpoint before x
    greater_checkpoints = [(e, d) for e,d in checkpoints if d > x ]
    if x == cp1[1]:
        return cp1[0]
    elif len(greater_checkpoints) == 0:
        return 0

    # checkpoint after x
    cp2 = greater_checkpoints[0]# checkpoint after x
    
    loss_rate, losses = get_loss_info((cp1, cp2, losses))
    
    # losses since last checkpoint
    stoch_loss_since_cp1 = sum([l[0] for l in losses if l[1] <= x])

    # (E at last cp) - (stoch losses since last cp) - (loss rate * distance from last cp)
    energy = cp1[0] - stoch_loss_since_cp1 - (x - cp1[1]) * loss_rate

    return energy

def plot_dEdx(hists, E_bins, points_labels, plotdir):

    colors = ['b', 'g', 'm']
    
    fig = plt.figure()

    for hist,label,color in itertools.izip(hists, points_labels, colors):
        print(label)
        
        w = hist.weights
        w[w == 0] = 1
        w2 = hist.weights2
        w2[w2 == 0] = 1

        x = hist.x_weighted / w
        y = hist.y_weighted / w
        x2 = hist.x2_weighted / w
        y2 = hist.y2_weighted / w

        stddev_by_bin = np.sqrt(y2 - y**2)

        error_of_mean_by_bin = stddev_by_bin * np.sqrt(w2) / w

        alt_err = np.where(y < error_of_mean_by_bin, 0, error_of_mean_by_bin)

        plt.errorbar(x, y, fmt=color+'o', label=label, yerr=[alt_err, error_of_mean_by_bin])
        plt.hist(x, bins=E_bins, weights=y, color=color, histtype='step')
    
    line = lambda x: 0.259*0.917+0.363*10**(-3)*0.917*x
    plt.plot(x,line(x), 'ro', linewidth=2.0, label='Expected total losses')
    plt.axis([1, 10**6, 10**(-1), 10**4])
    plt.legend(loc=2)
    plt.xlabel('Muon Energy (GeV)')
    plt.ylabel('dE/dx (GeV/m)')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(plotdir + 'dEdx.png')
    plt.close(fig)
    #plt.show()

def add_E_dEdx_points(losses, weights, checkpoints, mu_info, get_point, hist, sample_d = 10 ):
    # Don't go out of the simulation volume. Each muon has at least 2 checkponts (start, end), often a third where it exits simulation volume
    
    n_muons = 0
    for cps,loss_tuples,mu,weight in itertools.izip(checkpoints, losses, mu_info, weights):
        #if(mu[muon_p_energy] < 5000):
        #    continue
        #if(np.sqrt(mu[muon_p_pos_x]**2 + mu[muon_p_pos_y]**2 + mu[muon_p_pos_z]**2) > 500):
        #    continue
        point_E = []
        point_dEdx = []
        point_weight = []

        if(len(cps) == 4):
            if cps[1][0] == 0:
                min_cp = 0
            else:
                min_cp = 1

            if cps[2][0] == 0:
                max_cp = 3
            else:
                max_cp = 2
        else:
            raise

        min_range = cps[min_cp][1]
        max_range = cps[max_cp][1]
        new_cps = [cps[min_cp], cps[max_cp]]

        if max_range - min_range > 10000:
            print("Range greater than 10000m!")

        x1 = min_range
        x2 = x1 + sample_d
        while(x2 <= max_range):
            try:
                E, dEdx = get_point(x1, x2, new_cps, tuple(loss_tuples))
            except:
                print(x1, x2)
                print(new_cps)
                raise
            point_E.append(E)
            point_dEdx.append(dEdx)
            point_weight.append(weight)
            x1 += sample_d
            x2 += sample_d
        n_muons += 1
        #print("Have %d muons!" % n_muons)
        hist.add(point_E, point_dEdx, point_weight)

    return n_muons

def get_hists_from_dirs(indirs, E_bins, points_functions):
    # Get list of input pickle files from all input directories
    infiles = [file for indir in indirs for file in glob.glob('%s*.pkl' % indir)]
    infiles.sort()

    n_muons = 0

    hists = []
    for i in points_functions:
        hists.append(histogram(E_bins))

    # Loop over input files
    for infile in infiles[:15]:
        if os.stat(infile).st_size == 0:
            continue
        print(infile)
        # Load fractional losses for each muon
        inpickle = open(infile, 'rb')
        losses = pickle.load(inpickle)

        # Load weights for each muon
        weights = pickle.load(inpickle)

        # Load track energy checkpoints for each muon
        checkpoints = pickle.load(inpickle)
        
        mu_info = pickle.load(inpickle)
        #nu_info = np.array(pickle.load(inpickle))

        losses = np.array(losses)
        weights = np.array(weights)
        checkpoints = np.array(checkpoints)
        mu_info = np.array(mu_info)
        
        new_muons = 0
        for func,hist in itertools.izip(points_functions, hists):
            new_muons = add_E_dEdx_points(losses, weights, checkpoints, mu_info, func, hist)
        n_muons += new_muons

        print("Have %d muons!" % n_muons)

        inpickle.close()

    return hists

def get_info_from_file(infile):
    inpickle = open(infile, 'rb')
    hists = pickle.load(inpickle)
    E_bins = pickle.load(inpickle)

    inpickle.close()

    return (hists, E_bins)

def save_info_to_file(outfile, hists, E_bins):
    outpickle = open(outfile, 'wb')
    pickle.dump(hists, outpickle, -1)
    pickle.dump(E_bins, outpickle, -1)

    outpickle.close()
    
def get_total_losses_point(x1, x2, cps, loss_tuples):
    E_at_x1 = get_energy(x1, cps, loss_tuples)
    E_at_x2 = get_energy(x2, cps, loss_tuples)
    return (E_at_x1, abs((E_at_x1 - E_at_x2)) / (x2 - x1))
   
def get_stoch_losses_point(x1, x2, cps, loss_tuples):
    return (get_energy(x1, cps, loss_tuples), abs(sum([e for e,d,t in loss_tuples if t not in exclude and d >= x1 and d < x2]) / (x2 - x1)))

def get_mc_stoch_losses_point(x1, x2, cps, loss_tuples):
    return (get_energy(x1, cps, loss_tuples), abs(sum([e for e,d,t in loss_tuples if d >= x1 and d < x2]) / (x2 - x1)))

points_functions = [get_total_losses_point, get_stoch_losses_point, get_mc_stoch_losses_point]
points_labels = ['MC total losses', 'MC stochastic losses', 'MC stochastic losses w/ ionization']

#points_functions = [get_total_losses_point]
#points_labels = ['MC total losses']

# Create log energy bins
E_max = 1000000 # 1PeV
E_min = .000001
#E_bins = np.logspace(0, 6, 48+1)
E_bins = np.logspace(0, 6, 12+1)

info = None

if infile == '':
    if len(indirs):
        hists = get_hists_from_dirs(indirs, E_bins, points_functions)
        if outfile != '':
            save_info_to_file(outfile, hists, E_bins)
else:
    hists, E_bins = get_info_from_file(infile)

plot_dEdx(hists, E_bins, points_labels, plotdir)
