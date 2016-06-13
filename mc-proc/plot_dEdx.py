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

# Plots fractional muon energy loss as a fucntion of distance
# Takes fractional energy loss and position information from pickle files
# Pickle:
    # fractional_losses: [[(muon_0_fractional_loss_0, muon_0_track_distance_0), (m_0_fl_1, m_0_d_1), ...], [(m_1_fl_0, m_1_d_0), (m_1_fl_1, m_1_d_1), ...], ...]
    # weights: [muon_0_weight, muon_1_weight, ...]

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

# Define fucntion to get [1] item from list or tuple
_get1 = op.itemgetter(1)

@memodict
def get_loss_info(stuff):
    """
    Get the loss rate and losses between two energy checkpoints.
    Takes single tuple argument (checkpoint1, checkpoint2, losses) to allow fast memoization.
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

def get_bin_weights(points_by_entry, bins, weights):
    """
    Get weighted entries for each bin.
    Also return weighting information for each bin,
        to be used in later normalization and error calculation.
    """
    # List of points for each bin for each entry
    points_by_bin = np.array([[[(x, y) for x,y in points if x >= b1 and x < b2] for b1,b2 in itertools.izip(bins[:-1], bins[1:])] for points in points_by_entry])

    #weighted_y_by_bin = np.zeros((len(bins) - 1, 0)).tolist()
    ##y_weights = np.zeros((len(bins) - 1, 0)).tolist()
    #weighted_y_by_entry_by_bin = [[[(y, weight) for x,y in bin] for bin in points] for points,weight in itertools.izip(points_by_bin, weights)]
    #for i_entry in xrange(0, len(weighted_y_by_bin)):
    #    for i_bin in xrange(0, len(bins)-1):
    #        weighted_y_by_bin[i_bin] += weighted_y_by_entry_by_bin[i_entry][i_bin]
    #        #y_weights[i_bin] += [weights[i_entry]] * len(weighted_y_by_entry_by_bin[i_entry][i_bin])

    #weighted_x_by_bin = np.zeros((len(bins) - 1, 0)).tolist()
    ##x_weights = np.zeros((len(bins) - 1, 0)).tolist()
    #weighted_x_by_entry_by_bin = [[[(x, weight) for x,y in bin] for bin in points] for points,weight in itertools.izip(points_by_bin, weights)]
    #for i_entry in xrange(0, len(weighted_x_by_bin)):
    #    for i_bin in xrange(0, len(bins)-1):
    #        weighted_x_by_bin[i_bin] += weighted_x_by_entry_by_bin[i_entry][i_bin]
    #        #x_weights[i_bin] += [weights[i_entry]] * len(weighted_x_by_entry_by_bin[i_entry][i_bin])

    # Sum of y values for each bin for each entry
    y_sum_by_bin = np.array([[sum([y for x,y in bin]) for bin in points] for points in points_by_bin])
    
    # Sum of y**2 values for each bin in each entry
    y_sq_sum_by_bin = np.array([[sum([y*y for x,y in bin]) for bin in points] for points in points_by_bin])
    
    # Sum of x values for each bin for each entry
    x_by_bin = np.array([[[x for x,y in bin] for bin in points] for points in points_by_bin])
    x_sum_by_bin = np.array([[sum([x for x,y in bin]) for bin in points] for points in points_by_bin])
    
    # Number of points for each bin for each entry
    len_by_bin = np.array([[len(bin) for bin in points] for points in points_by_bin])

    # Sum of weights of each bin
    bin_weights = np.dot(weights, len_by_bin)

    # Sum of weights**2 for each bin
    bin_sq_weights = np.dot(weights*weights, len_by_bin)

    weighted_y_sum = np.dot(weights, y_sum_by_bin)
    weighted_y_sq_sum = np.dot(weights, y_sq_sum_by_bin)
    weighted_x_sum = np.dot(weights, x_sum_by_bin)

    #return (weighted_x_sum, weighted_y_sum, bin_weights, weighted_y_sq_sum, bin_sq_weights, weighted_x_by_bin, weighted_y_by_bin)
    return (weighted_x_sum, weighted_y_sum, bin_weights, weighted_y_sq_sum, bin_sq_weights)

def plot_dEdx(infos, E_bins, points_labels, plotdir):

    colors = ['b', 'g', 'm']
    
    for info,label,color in itertools.izip(infos, points_labels, colors):
        print(label)

        #file_weighted_E, file_weighted_dEdx, file_bin_weights, file_weighted_dEdx_sq, file_bin_sq_weights, weighted_x_by_file_by_bin, weighted_y_by_file_by_bin = info
        file_weighted_E, file_weighted_dEdx, file_bin_weights, file_weighted_dEdx_sq, file_bin_sq_weights = info
        weighted_x_by_bin = np.zeros((len(E_bins) - 1, 0)).tolist()
        for i_file in xrange(0, len(weighted_x_by_file_by_bin)):
            for i_bin in xrange(0, len(E_bins)-1):
                weighted_x_by_bin[i_bin] += weighted_x_by_file_by_bin[i_file][i_bin]

        weighted_y_by_bin = np.zeros((len(E_bins) - 1, 0)).tolist()
        for i_file in xrange(0, len(weighted_y_by_file_by_bin)):
            for i_bin in xrange(0, len(E_bins)-1):
                weighted_y_by_bin[i_bin] += weighted_y_by_file_by_bin[i_file][i_bin]

        #for contents,bin1,bin2 in itertools.izip(weighted_x_by_bin, E_bins[:-1], E_bins[1:]):
        #    points = [x for x,w in contents]
        #    weights = np.array([w for x,w in contents])
        #    weights = weights / sum(weights)

        #    non_zero_points = [p for p in points if p > 0]

        #    fig = plt.figure()
        #    if len(points):
        #        fine_bins = np.logspace(np.floor(np.log10(min(non_zero_points))), np.ceil(np.log10(max(non_zero_points))), 20+1)
        #        plt.hist(points, weights=weights, bins=fine_bins, color=color, histtype='step')
        #    else:
        #        plt.plot([])
        #    title = label + ':E in %fGeV-%fGeV bin' % (bin1, bin2)

        #    print(title + ': %d' % len(points))
        #    print('len: %d' % len(points))
        #    if(len(points)):
        #        print('min: ', min(contents, key=op.itemgetter(0)))
        #        print('max: ', max(contents, key=op.itemgetter(0)))
        #    print
        #   
        #    plt.title(title)
        #    plt.xscale('log')
        #    plt.yscale('log')
        #    plt.ylim([10**(-7), 1])
        #    plt.xlabel('Muon Energy (GeV)')
        #    plt.ylabel('Counts')
        #    plt.savefig(plotdir + re.sub(' ', '_', title.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}')) + '.png')
        #    plt.close(fig)
        #    #plt.show()

        #for contents,bin1,bin2 in itertools.izip(weighted_y_by_bin, E_bins[:-1], E_bins[1:]):
        #    points = [x for x,w in contents]
        #    weights = np.array([w for x,w in contents])
        #    weights = weights / sum(weights)

        #    non_zero_points = [p for p in points if p > 0]

        #    fig = plt.figure()
        #    if len(points) and len(non_zero_points):
        #        fine_bins = np.logspace(np.floor(np.log10(min(non_zero_points))), np.ceil(np.log10(max(non_zero_points))), 20+1)
        #        plt.hist(points, weights=weights, bins=fine_bins, color=color, histtype='step')
        #    else:
        #        plt.plot([])
        #    title = label + ':dE/dx in %fGeV-%fGeV bin' % (bin1, bin2)
        #    
        #    print(title)
        #    print('len: %d' % len(points))
        #    if(len(points)):
        #        print('min: ', min(contents, key=op.itemgetter(0)))
        #        print('max: ', max(contents, key=op.itemgetter(0)))
        #    print
        #    
        #    plt.title(title)
        #    plt.xscale('log')
        #    plt.yscale('log')
        #    plt.ylim([10**(-7), 1])
        #    plt.xlabel('Muon Energy Loss (GeV/m)')
        #    plt.ylabel('Counts')
        #    plt.savefig(plotdir + re.sub(' ', '_', title.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}')) + '.png')
        #    plt.close(fig)
        #    #plt.show()
    colors = ['b', 'g', 'm']
  
    colors = ['b', 'g', 'm']
   
    fig = plt.figure()

    for info,label,color in itertools.izip(infos, points_labels, colors):
        #file_weighted_E, file_weighted_dEdx, file_bin_weights, file_weighted_dEdx_sq, file_bin_sq_weights, weighted_x_by_file_by_bin, weighted_y_by_file_by_bin = info
        file_weighted_E, file_weighted_dEdx, file_bin_weights, file_weighted_dEdx_sq, file_bin_sq_weights = info
        bin_weights = sum(file_bin_weights)
        bin_weights = np.array([weight if weight > 0 else 1 for weight in bin_weights])
        dEdx_by_bin = sum(file_weighted_dEdx) / bin_weights
        dEdx_sq_by_bin = sum(file_weighted_dEdx_sq) / bin_weights
        stddev_by_bin = np.sqrt(dEdx_sq_by_bin - dEdx_by_bin**2)
        E_by_bin = sum(file_weighted_E) / bin_weights

        error_of_mean_by_bin = stddev_by_bin * np.sqrt(sum(file_bin_sq_weights)) / bin_weights

        alt_err = error_of_mean_by_bin.copy()
        alt_err[dEdx_by_bin < alt_err] = 0

        plt.errorbar(E_by_bin, dEdx_by_bin, fmt=color+'o', label=label, yerr=[alt_err, error_of_mean_by_bin])
        plt.hist(E_by_bin, bins=E_bins, weights=dEdx_by_bin, color=color, histtype='step')
    
    line = lambda x: 0.259*0.917+0.363*10**(-3)*0.917*x
    plt.plot(E_by_bin,line(E_by_bin), 'ro', linewidth=2.0, label='Expected total losses')
    plt.axis([1, 10**6, 10**(-1), 10**4])
    plt.legend(loc=2)
    plt.xlabel('Muon Energy (GeV)')
    plt.ylabel('dE/dx (GeV/m)')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(plotdir + 'dEdx.png')
    plt.close(fig)
    #plt.show()

def get_E_dEdx_points(losses, weights, checkpoints, mu_info, get_point, sample_d = 10 ):
    # Don't go out of the simulation volume. Each muon has at least 2 checkponts (start, end), often a third where it exits simulation volume
    
    #max_range = [cps[-2][1] if len(cps) > 2 else cps[-1][1] for cps in checkpoints]
    points = []
    #for cps,loss_tuples,r in itertools.izip(checkpoints, losses, max_range):
    for cps,loss_tuples,mu in itertools.izip(checkpoints, losses, mu_info):
        min_range = 0
        max_range = 0
        if(is_in_sim_vol(mu)):
            min_range = 0
            max_range = cps[1][1] # Can either have cps (start, exit, end) or (start, end)
        else:
            if(len(cps) == 2):
                continue
            elif(len(cps) == 3):
                min_range = cps[1][1]
                max_range = cps[-1][1]
            elif(len(cps) == 4):
                min_range = cps[1][1]
                max_range = cps[2][1]
        points_for_muon = []
        x1 = min_range
        x2 = sample_d
        while(x2 <= max_range):
            points_for_muon.append(get_point(x1, x2, cps, tuple(loss_tuples)))
            x1 += sample_d
            x2 += sample_d
        points.append(points_for_muon)
    return points 

def get_info_from_dirs(indirs, E_bins, points_functions):
    # Get list of input pickle files from all input directories
    infiles = [file for indir in indirs for file in glob.glob('%s*.pkl' % indir)]
    infiles.sort()

    # dEdx plot
    file_weighted_E = []
    file_weighted_dEdx = []
    file_bin_weights = []

    file_weighted_stoch_E = []
    file_weighted_stoch_dEdx = []
    file_stoch_bin_weights = []

    n_muons = 0

    #n_returns = 6

    #infos = np.zeros(shape=(len(points_functions), n_returns, 0)).tolist()
    infos = None

    # Loop over input files
    for infile in infiles[:6]:
        if os.stat(infile).st_size == 0:
            continue
        print(infile)
        # Load fractional losses for each muon
        inpickle = open(infile, 'rb')
        losses = pickle.load(inpickle)
        n_muons += len(losses)

        # Load weights for each muon
        weights = pickle.load(inpickle)

        # Load track energy checkpoints for each muon
        checkpoints = pickle.load(inpickle)
        
        mu_info = pickle.load(inpickle)
        #nu_info = np.array(pickle.load(inpickle))

        #is_in_sim_mask = [is_in_sim_vol(m) for m in mu_info]
        #losses = [loss_tuples for loss_tuples,b in itertools.izip(losses, is_in_sim_mask) if b]
        #weights = [w for w,b in itertools.izip(weights, is_in_sim_mask) if b]
        #checkpoints = [cps for cps,b in itertools.izip(checkpoints, is_in_sim_mask) if b]
        #mu_info = [m for m,b in itertools.izip(mu_info, is_in_sim_mask) if b]

        losses = np.array(losses)
        weights = np.array(weights)
        checkpoints = np.array(checkpoints)
        mu_info = np.array(mu_info)
        
        points_entries = [get_E_dEdx_points(losses, weights, checkpoints, mu_info, func) for func in points_functions]
        info_entries = [get_bin_weights(points, E_bins, weights) for points in points_entries]

        if infos == None:
            infos = np.zeros(shape=(len(points_functions), len(info_entries[0]), 0)).tolist()

        for i_info in xrange(0, len(infos)):
            for i_list in xrange(0, len(infos[i_info])):
                infos[i_info][i_list].append(info_entries[i_info][i_list])

        inpickle.close()

    return infos

def get_info_from_file(infile):
    inpickle = open(infile, 'rb')
    infos = pickle.load(inpickle)
    E_bins = pickle.load(inpickle)

    inpickle.close()

    return (infos, E_bins)

def save_info_to_file(outfile, infos, E_bins):
    outpickle = open(outfile, 'wb')
    pickle.dump(infos, outpickle, -1)
    pickle.dump(E_bins, outpickle, -1)

    outpickle.close()
    
def get_total_losses_point(x1, x2, cps, loss_tuples):
    #return (get_energy((x2+x1)/2, cps, loss_tuples), abs((get_energy(x1, cps, loss_tuples) - get_energy(x2, cps, loss_tuples)) / (x2 - x1)))
    E_at_x1 = get_energy(x1, cps, loss_tuples)
    E_at_x2 = get_energy(x2, cps, loss_tuples)
    return (E_at_x1, abs((E_at_x1 - E_at_x2)) / (x2 - x1))
   
def get_stoch_losses_point(x1, x2, cps, loss_tuples):
    #return (get_energy((x2+x1)/2, cps, loss_tuples), abs(sum([e for e,d,t in loss_tuples if t not in exclude and d >= x1 and d < x2]) / (x2 - x1)))
    return (get_energy(x1, cps, loss_tuples), abs(sum([e for e,d,t in loss_tuples if t not in exclude and d >= x1 and d < x2]) / (x2 - x1)))

def get_mc_stoch_losses_point(x1, x2, cps, loss_tuples):
    return (get_energy(x1, cps, loss_tuples), abs(sum([e for e,d,t in loss_tuples if d >= x1 and d < x2]) / (x2 - x1)))

points_functions = [get_total_losses_point, get_stoch_losses_point, get_mc_stoch_losses_point]
points_labels = ['MC total losses', 'MC stochastic losses', 'MC stochastic losses w/ ionization']

# Create log energy bins
E_max = 1000000 # 1PeV
E_min = .000001
#E_bins = np.logspace(0, 6, 48+1)
E_bins = np.logspace(0, 6, 12+1)

info = None

if infile == '':
    if len(indirs):
        infos = get_info_from_dirs(indirs, E_bins, points_functions)
        if outfile != '':
            save_info_to_file(outfile, infos, E_bins)
else:
    infos, E_bins = get_info_from_file(infile)

plot_dEdx(infos, E_bins, points_labels, plotdir)
