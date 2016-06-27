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
        self.size = len(bins) - 1
        size = self.size
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
        
    def get_w(self):
        w = self.weights[:]
        w[w == 0] = 1
        return w
    def get_w2(self):
        w2 = self.weights2[:]
        w2[w2 == 0] = 1
        return w2
    def get_x(self):
        return self.x_weighted / self.get_w()
    def get_y(self):
        return self.y_weighted / self.get_w()
    def get_x2(self):
        return self.x2_weighted / self.get_w()
    def get_y2(self):
        return self.y2_weighted / self.get_w()
    def stddev(self, x, x2):
        variance = np.zeros(self.size)
        good_var = np.logical_not(np.logical_or(np.isclose(x2, x**2, rtol=1e-05, atol=1e-09), x2 < x**2))
        variance[good_var] = x2[good_var] - x[good_var]**2
        return np.sqrt(variance)
    def get_x_stddev(self):
        return self.stddev(self.get_x(), self.get_x2())
    def get_y_stddev(self):
        return self.stddev(self.get_y(), self.get_y2())
    def get_x_stddev_of_mean(self):
        return self.get_x_stddev() * np.sqrt(self.get_w2()) / self.get_w()
    def get_y_stddev_of_mean(self):
        return self.get_y_stddev() * np.sqrt(self.get_w2()) / self.get_w()

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

def get_energy(x, checkpoints, losses, inclusive=True):
    """
    Get the energy of a muon track at a point x.
    Given energy checkpoints and losses along track.
    """
    x = max(0, x) # Return zero for negative positions
    checkpoints = sorted(checkpoints, key=_get1) # Sort by distance
    lesser_checkpoints = [(e, d) for e,d in checkpoints if d <= x ] # checkpoints before x
    if len(lesser_checkpoints) == 0:
        print("x: %f" % x)
        print("Checkpoints: ", checkpoints)
        raise ValueError("There are no checkpoints before x!")
    cp1 = lesser_checkpoints[-1]
    greater_checkpoints = [(e, d) for e,d in checkpoints if d > x ]
    if x == cp1[1]:
        return cp1[0]
    elif len(greater_checkpoints) == 0:
        return 0

    # checkpoint after x
    cp2 = greater_checkpoints[0] # checkpoint after x
    
    loss_rate, losses = get_loss_info((cp1, cp2, losses))
    
    # losses since last checkpoint
    if inclusive:
        stoch_loss_since_cp1 = sum([l[0] for l in losses if l[1] <= x])
    else:
        stoch_loss_since_cp1 = sum([l[0] for l in losses if l[1] < x])

    # (E at last cp) - (stoch losses since last cp) - (loss rate * distance from last cp)
    energy = cp1[0] - stoch_loss_since_cp1 - (x - cp1[1]) * loss_rate

    return energy

def plot(hists, dist_bins, points_labels, plotdir):

    colors = ['b', 'g', 'm', 'c']
    
    fig = plt.figure()

    for hist,label,color in itertools.izip(hists, points_labels, colors):
        print(label)
        
        #x = hist.get_x()
        x = (hist.bins[:-1] + hist.bins[1:]) / 2
        y = hist.get_y()

        error_of_mean_by_bin = hist.get_y_stddev_of_mean()

        alt_err = np.where(y < error_of_mean_by_bin, 0, error_of_mean_by_bin)

        plt.errorbar(x, y, fmt=color+'.', label=label, yerr=[alt_err, error_of_mean_by_bin])
        try:
            plt.hist(x, bins=dist_bins, weights=y, color=color, histtype='step')
        except:
            print(label)
            print("x: ", x)
            print("y: ", y)
            raise

        magic_ice_const = 0.917

        x2 = max(hist.bins)
        dxs = np.unique(hist.bins[1:].astype(int) - hist.bins[:-1].astype(int))
        select_dx = np.vectorize(lambda x: dxs[next(itertools.dropwhile(lambda xx: xx[1] > x, enumerate(hist.bins)))[0]])
        c_func = lambda b, dx, x2: np.log(b*dx/(1-np.exp(-b*x2)))

        fit_func = lambda x, b: np.exp(-b*x+c_func(b, select_dx(x), x2))
        #fit_func = lambda x, b, a: np.exp(x*b*0.917+a*0.917)
        #fit_func = lambda x, b, a: x*b*0.917+a*0.917
        # Fit the histogram datapoints to an exponential
        #fit = scipy.optimize.curve_fit(fit_func, x, y, [-4.68*10**(-4), -2.15], error_of_mean_by_bin)
        fit = scipy.optimize.curve_fit(fit_func, x, y, [4.68*10**(-4)], error_of_mean_by_bin)
        #print('Exponential fit: y=Exp[%f * x + %f]' % (fit[0][0]*0.917, fit[0][1]*0.917))
        print('Exponential fit: y=Exp[%.9f * x + %f]' % (-fit[0][0], c_func(fit[0][0], sum(dxs)/len(dxs), x2)))

        print('L=%f' % ((fit[0][0] / magic_ice_const)**(-1)))
        
        plt.plot(x, fit_func(x, fit[0][0]), 'r')
    
        #line = lambda x: 0.259*0.917+0.363*10**(-3)*0.917*x
        #plt.plot(x,line(x), 'r.', linewidth=2.0, label='Expected total losses')
        #plt.axis([1, 10**6, 10**(-4), 10**4])
        plt.legend(loc=3)
        plt.xlabel('Track Distance (m)')
        plt.ylabel('Fractional Energy Loss')
        plt.title(label)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.savefig(plotdir + 'fractional_energy_loss_' + re.sub(' ', '_', label.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}')) + '.png')
        plt.clf()
        plt.close(fig)
    #plt.show()

def add_points(losses, weights, checkpoints, mu_info, get_point, hist):
    n_muons = 0
    for cps,loss_tuples,mu,weight in itertools.izip(checkpoints, losses, mu_info, weights):
        if(mu[muon_p_energy] < 5000):
            continue
        #if(np.sqrt(mu[muon_p_pos_x]**2 + mu[muon_p_pos_y]**2 + mu[muon_p_pos_z]**2) > 500):
        #    continue
        point_x = []
        point_dE = []
        point_weight = []

        if(len(cps) == 4):
            if cps[1][0] <= 0:
                min_cp = 0
            else:
                min_cp = 1

            if cps[2][0] <= 0:
                max_cp = 3
            else:
                max_cp = 2
        else:
            raise

        min_range = cps[min_cp][1]
        max_range = cps[max_cp][1]

        # Skip short tracks
        if(max_range - min_range < max(hist.bins)):
            continue

        new_cps = (tuple(cps[min_cp]), tuple(cps[max_cp]))

        if max_range - min_range > 10000:
            print("Range greater than 10000m!")

        next_losses = sorted(loss_tuples, key=_get1)[:]
        loss_tuples = tuple(loss_tuples)
        last_E = new_cps[0][0]
        break_next = False

        #print "Range: ", (min_range, max_range)
        for x1,x2 in itertools.izip(hist.bins[:-1], hist.bins[1:]):
            x1 += min_range
            x2 += min_range
            if(x1 > max_range):
                x = x1
                dE = 0
            else:
                if(x2 > max_range):
                    x2 = max_range
                x, dE = get_point((x1, x2, new_cps, loss_tuples))
            #print(x, dE)
            point_x.append(x)
            point_dE.append(dE)
            point_weight.append(weight)
            #if(x2 == max_range):
            #    break
        total_dE = sum(point_dE)
        if(total_dE == 0):
            continue
        point_frac_dE = [dE / total_dE for dE in point_dE]
        point_x = [x - min_range for x in point_x]

        n_muons += 1
        #print("Have %d muons!" % n_muons)
        #print(point_E, point_dEdx, point_weight)
        hist.add(point_x, point_frac_dE, point_weight)

        if not n_muons % 1000:
            print("%d muons in file" % n_muons)
        
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
    for infile in infiles:
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
            new_muons = add_points(losses, weights, checkpoints, mu_info, func, hist)
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
    
def get_total_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    E_at_x1 = get_energy(x1, cps, loss_tuples)
    E_at_x2 = get_energy(x2, cps, loss_tuples)
    return (x1, abs((E_at_x1 - E_at_x2)))
   
def get_stoch_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    return (x1, abs(sum([e for e,d,t in loss_tuples if t not in exclude and d > x1 and d <= x2])))

def get_mc_stoch_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    return (x1, abs(sum([e for e,d,t in loss_tuples if d > x1 and d <= x2])))

def get_cont_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    E_at_x1 = get_energy(x1, cps, loss_tuples)
    E_at_x2 = get_energy(x2, cps, loss_tuples)
    return (x1, (abs((E_at_x1 - E_at_x2)) - abs(sum([e for e,d,t in loss_tuples if d > x1 and d <= x2]))))

points_functions = [get_total_losses_point, get_stoch_losses_point, get_mc_stoch_losses_point, get_cont_losses_point]
points_labels = ['MC total losses', 'MC stochastic losses', 'MC stochastic losses w/ ionization', 'MC continuous losses']

#points_functions = [get_total_losses_point]
#points_labels = ['MC total losses']

# Create log energy bins
E_max = 1000000 # 1PeV
E_min = .000001
#E_bins = np.logspace(0, 6, 48+1)
#E_bins = np.logspace(0, 6, 12+1)
dist_bins = np.linspace(0, 600, 10+1)

info = None

if infile == '':
    if len(indirs):
        hists = get_hists_from_dirs(indirs, dist_bins, points_functions)
        if outfile != '':
            save_info_to_file(outfile, hists, dist_bins)
else:
    hists, dist_bins = get_info_from_file(infile)

plot(hists, dist_bins, points_labels, plotdir)
