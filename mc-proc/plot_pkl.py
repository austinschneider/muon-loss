import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize

from icecube import dataclasses

import os
import pickle
import cPickle
import glob
import argparse
import itertools
import re
import operator as op

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

def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__

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

# Useful numbers for accessing muon information
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
    
    def add(self, x, y, weights, bin=None):
        x = np.array(x)
        y = np.array(y)
        weights = np.array(weights)

        if bin:
            bin = np.array(bin)
            weights_by_bin = np.bincount(bin, minlength=self.size, weights=weights)
            weights2_by_bin = np.bincount(bin, minlength=self.size, weights=weights**2)
            x_weighted_by_bin = np.bincount(bin, minlength=self.size, weights=x * weights)
            y_weighted_by_bin = np.bincount(bin, minlength=self.size, weights=y * weights)
            x2_weighted_by_bin = np.bincount(bin, minlength=self.size, weights=x**2 * weights)
            y2_weighted_by_bin = np.bincount(bin, minlength=self.size, weights=y**2 * weights)
        else:
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

# Define fucntion to get item from list or tuple
_get0 = op.itemgetter(0)
_get1 = op.itemgetter(1)

def get_valid_checkpoints(cps):
    """
    Takes list of checkpoints
    Assumes that the first and last checkpoints are the track begin and end respectively
    Returns only checkpoints from the original list that are valid
        Invalid checkpoints are those that have energy <= 0 (excluding the track begin and end)
    """
    track_cps = cps[1:-1]
    new_cps = [cps[0]] + [cp for cp in track_cps if cp[0] > 0] + [cps[-1]]
    return tuple(sorted(new_cps, key=_get1))

def add_loss_sum(losses, checkpoints):
    """
    Takes list of losses and checkpoints for a single track
    Assumes that all the checkpoints are valid
    Returns a list of losses with the loss sums added as the last element of each loss
    """
    next_dist = 0
    total = 0
    losses = sorted(list(losses), key=_get1)
    for j in xrange(len(losses)):
        if losses[j][1] >= next_dist:
            next_dist = next(itertools.dropwhile(lambda cp: cp[1] <= losses[j][1], checkpoints), (None, np.inf))[1]
            total = 0
        total += losses[j][0]
        losses[j] = tuple(list(losses[j]) + [total])
    return tuple(losses)

def get_track_range(cps):                                                                                                                                                                                   
    """ 
    Takes list of checkpoints
    Returns range of track inside simulation volume
    """
    track_cps = cps[1:-1]
    if track_cps[0][0] <= 0:
        min_range = cps[0][1]
    else:
        min_range = track_cps[0][1]
    if track_cps[-1][0] <= 0:
        max_range = cps[-1][1]
    else:
        max_range = track_cps[-1][1]

    return (min_range, max_range)

@memodict
def get_loss_info(stuff):
    """
    Get the loss rate and losses between two energy checkpoints.
    Takes single tuple argument (checkpoint1, checkpoint2, losses) to allow fast memoization.
    A checkpoint has the structure: (energy, distance along track)
    """
    cp1, cp2, losses, has_sum = stuff
    losses = sorted(losses, key=_get1) # Sort by distance 
    losses = [l for l in losses if (l[1] > cp1[1]) and (l[1] < cp2[1])]

    if has_sum:
        if len(losses):
            total_stochastic_loss = losses[-1][3]
        else:
            total_stochastic_loss = 0
    else:
        total_stochastic_loss = sum([l[0] for l in losses])
    loss_rate = (cp1[0] - cp2[0] - total_stochastic_loss) / (cp2[1] - cp1[1])

    return (loss_rate, losses)

def get_bounding_elements(x, l, key=lambda elem: elem, sort=False):
    if sort:
        l = sorted(l, key=key)
    return (next(itertools.dropwhile(lambda elem: key(elem) > x, reversed(l)), None),
            next(itertools.dropwhile(lambda elem: key(elem) < x, l), None))

def get_energy(x, checkpoints, loss_tuples, inclusive=True, has_sum=True):
    """
    Get the energy of a muon track at a point x.
    Given energy checkpoints and losses along track.
    """
    # Get the checkpoints on either side of x, search by distance
    cp1, cp2 = get_bounding_elements(x, checkpoints, _get1)

    # If the checkpoints are the same then x has the same distance as the checkpoint
    if cp1 == cp2:
        return cp1[0]

    # Return 0 for regions in which we don't have enough information
    if not cp1:
        return 0
    if not cp2:
        return 0

    # Get the loss rate and losses between the checkpoints
    loss_rate, losses = get_loss_info((cp1, cp2, loss_tuples, has_sum))
    
    # Get the sum of losses between x and the checkpoint before x
    if inclusive:
        if has_sum:
            i_loss_before_x = next(itertools.dropwhile(lambda loss: loss[1][1] <= x, enumerate(losses)), [len(losses)])[0] - 1
            if i_loss_before_x < 0:
                stoch_loss_since_cp1 = 0
            else:
                stoch_loss_since_cp1 = losses[i_loss_before_x][3]
        else:
            stoch_loss_since_cp1 = sum([l[0] for l in losses if l[1] <= x])
    else:
        if has_sum:
            i_loss_before_x = next(itertools.dropwhile(lambda loss: loss[1][1] < x, enumerate(losses)), [len(losses)])[0] - 1
            if i_loss_before_x < 0:
                stoch_loss_since_cp1 = 0
            else:
                stoch_loss_since_cp1 = losses[i_loss_before_x][3]
        else:
            stoch_loss_since_cp1 = sum([l[0] for l in losses if l[1] < x])

    # (E at last cp) - (stoch losses since last cp) - (loss rate * distance from last cp)
    energy = cp1[0] - stoch_loss_since_cp1 - (x - cp1[1]) * loss_rate

    #if(has_sum):
    #    other_energy = get_energy(x, checkpoints, loss_tuples, inclusive, False)
    #    if not (energy == other_energy or np.isclose(energy, other_energy, rtol=1e-05, atol=1e-09)):
    #        print("x: ", x)
    #        print("cp1: ", cp1)
    #        print("cp2: ", cp2)
    #        print("inclusive: ", inclusive)
    #        print("has_sum: ", has_sum)
    #        raise ValueError("Energy calculations do not match up! Energy: %f ; Other Energy: %f" % (energy, other_energy))

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

        #alt_err = np.where(y < error_of_mean_by_bin, 0, error_of_mean_by_bin)

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

def add_points(losses, weights, checkpoints, mu_info, points_functions, hists, E_bins, sample_d = 10, sample_E = 10):
    """
    Add points to the histogram for each muon with appropriate weighting
    """
    E_bins = sorted(E_bins)[:]
    n_muons = 0
    
    for cps,loss_tuples,mu,weight in itertools.izip(checkpoints, losses, mu_info, weights):
        # Make some cuts
        if(mu[muon_p_energy] < 5000):
            continue
        #if(np.sqrt(mu[muon_p_pos_x]**2 + mu[muon_p_pos_y]**2 + mu[muon_p_pos_z]**2) > 500):
        #    continue

        # Values to accumulate
        point_x = np.zeros((len(hists), 0)).tolist()
        point_dE = np.zeros((len(hists), 0)).tolist()
        point_frac_dE = np.zeros((len(hists), 0)).tolist()
        point_weight = np.zeros((len(hists), 0)).tolist()

        # We should always have 5 checkpoints
        if len(cps) != 5:
            raise ValueError("There are %d checkpoints instead of 5." % len(cps))
        
        min_range, max_range = get_track_range(cps)        
        new_cps = get_valid_checkpoints(cps)

        track_cps = cps[1:-1]
    
        starts_outside_simvol = track_cps[0][0] > 0
        if(starts_outside_simvol and track_cps[0][0] < 5000):
            continue

        # Sanity check on the segment of track we are considering
        if max_range - min_range > 10000:
            print("Range greater than 10000m!")

        if max_range - min_range < max(E_bins):
            continue

        # Keep track of losses not covered already
        next_losses = [loss for loss in sorted(loss_tuples, key=_get1) if loss[1] >= min_range and loss[1] <= max_range]
        
        # Convert to tuple for memoization
        loss_tuples = tuple(loss_tuples)

        for i in xrange(len(hists)):
            for x1,x2 in itertools.izip(hist.bins[:-1], hist.bins[1:]):
                x1 += min_range
                x2 += min_range
                if(x1 > max_range):
                    x = x1
                    dE = 0
                else:
                    if(x2 > max_range):
                        x2 = max_range
                    x, dE = points_functions[i]((x1, x2, new_cps, loss_tuples))
                #print(x, dE)
                point_x[i].append(x)
                point_dE[i].append(dE)
                point_weight[i].append(weight)
                #if(x2 == max_range):
                #    break
            total_dE = sum(point_dE)
            if(total_dE == 0):
                continue
            point_frac_dE[i] = [dE / total_dE for dE in point_dE[i]]
            point_x[i] = [x - min_range for x in point_x]

            hists[i].add(point_x[i], point_frac_dE[i], point_weight[i])

        n_muons += 1

        if not n_muons % 1000:
            print("%d muons in file" % n_muons)

        # Clear the memoization dictionary for get_loss_info since we are done with the muon
        get_loss_info.__self__.clear()
        
    return n_muons

def get_hists_from_dirs(indirs, E_bins, points_functions):
    """
    Compute histograms from pickle contained the list of input directories
    """
    # Get list of input pickle files from all input directories
    infiles = [file for indir in indirs for file in glob.glob('%s*.pkl' % indir)]
    infiles.sort()

    n_muons = 0

    # Create a histogram object for each function
    hists = []
    for i in points_functions:
        hists.append(histogram(E_bins))

    # Loop over input files
    for infile in infiles:
        if os.stat(infile).st_size == 0:
            continue
        print(infile)

        # Load losses for each muon
        inpickle = open(infile, 'rb')
        losses = cPickle.load(inpickle)

        # Load weights for each muon
        weights = cPickle.load(inpickle)

        # Load track energy checkpoints for each muon
        checkpoints = cPickle.load(inpickle)
        
        # Load muon and neutrino information
        mu_info = cPickle.load(inpickle)
        #nu_info = np.array(pickle.load(inpickle))

        # Preemptively calculate the sum of losses since the last valid checkpoint and append the information to the loss
        # Offers drastic performance improvement
        for i in xrange(len(losses)):
            losses[i] = add_loss_sum(losses[i], get_valid_checkpoints(checkpoints[i]))
        
        # Add points to the histograms using the points_functions
        n_muons += add_points(losses, weights, checkpoints, mu_info, points_functions, hists, E_bins)

        # Get rid of memory heavy references
        losses = None
        weights = None
        checkpoints = None
        mu_info = None
        nu_info = None

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
    return (x1, abs(sum([loss[0] for loss in loss_tuples if loss[2] not in exclude and loss[1] > x1 and loss[1] <= x2])))

def get_mc_stoch_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    return (x1, abs(sum([loss[0] for loss in loss_tuples if loss[1] > x1 and loss[1] <= x2]) / (x2 - x1)))

def get_cont_losses_point(stuff):
    x1, x2, cps, loss_tuples = stuff
    E_at_x1 = get_energy(x1, cps, loss_tuples)
    E_at_x2 = get_energy(x2, cps, loss_tuples)
    return (x1, (abs((E_at_x1 - E_at_x2)) - abs(sum([loss[0] for loss in loss_tuples if loss[1] > x1 and loss[1] <= x2]))) / (x2 - x1))

points_functions = [get_total_losses_point, get_stoch_losses_point, get_mc_stoch_losses_point, get_cont_losses_point]
points_labels = ['MC total losses', 'MC stochastic losses', 'MC stochastic losses w/ ionization', 'MC continuous losses']

#points_functions = [get_total_losses_point]
#points_labels = ['MC total losses']

dist_bins = np.linspace(0, 600, 10+1)

if infile == '':
    if len(indirs):
        hists = get_hists_from_dirs(indirs, dist_bins, points_functions)
        if outfile != '':
            save_info_to_file(outfile, hists, dist_bins)
else:
    hists, dist_bins = get_info_from_file(infile)

plot(hists, dist_bins, points_labels, plotdir)

