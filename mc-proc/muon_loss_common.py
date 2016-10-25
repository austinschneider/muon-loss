import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize

from icecube import dataclasses
import icetray
from I3Tray import *
from icecube import icetray, dataio
from icecube import dataclasses
from icecube import tableio, hdfwriter
from icecube import simclasses
from icecube import NewNuFlux
from icecube.icetray import I3Units
from icecube.weighting.weighting import from_simprod

import os
import pickle
import cPickle
import json
import glob
import argparse
import itertools
import re
import operator as op
import uuid
import shutil
import Queue
import multiprocessing
import threaded

# Define fucntion to get item from list or tuple
_get0 = op.itemgetter(0)
_get1 = op.itemgetter(1)

def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__

def get_particle_number(arg):
    """ Get the internal icecube particle number by its string name """
    if hasattr(arg, '__iter__'):
        s_list = arg
    elif type(arg) == str:
        s_list = [arg]
    else:
        raise ValueError("Argument is not a string or sequence")
    for i in xrange(len(s_list)):
        if type(s_list[i]) != str:
            raise ValueError("arg[%d] is not a string" % i)

    # Create dictionary to get particle type numbers by name
    particle_dict = {}
    p = dataclasses.I3Particle()
    for attr in dir(p.ParticleType):
        if not callable(attr) and not attr.startswith("__"):
            try:
                particle_dict[attr.lower()]=int(getattr(p.ParticleType, attr))
            except:
                pass
    n_list = [particle_dict[s.lower()] for s in s_list]
    return n_list

# Useful numbers for accessing muon information
muon_p_energy = 0
muon_p_pos_x = 1
muon_p_pos_y = 2
muon_p_pos_z = 3
muon_p_dir_zenith = 4
muon_p_dir_azimuth = 5
muon_p_length = 6

def add_losses_to_frame(frame, losses, has_sum=True):
    """
    Add loss information in a frame
    """
    loss_e = [loss[0] for loss in losses]
    loss_dist = [loss[1] for loss in losses]
    loss_type = [int(loss[2]) for loss in losses]
    if has_sum:
        loss_sum = [loss[3] for loss in losses]

    frame['MELLossesE'] = dataclasses.I3VectorDouble(loss_e)
    frame['MELLossesDist'] = dataclasses.I3VectorDouble(loss_dist)
    frame['MELLossesType'] = dataclasses.I3VectorInt(loss_type)
    if has_sum:
        frame['MELLossesSum'] = dataclasses.I3VectorDouble(loss_sum)

def get_losses_from_frame(frame, has_sum=True):
    """
    Get loss information from a frame
    """
    losses = []
    if has_sum:
        iterator = itertools.izip(frame['MELLossesE'], frame['MELLossesDist'], frame['MELLossesType'], frame['MELLossesSum'])
        for e, dist, type, sum in iterator:
            losses.append((e, dist, type, sum))
    else:
        iterator = itertools.izip(frame['MELLossesE'], frame['MELLossesDist'], frame['MELLossesType'])
        for e, dist, type in iterator:
            losses.append((e, dist, type))
    return tuple(losses)

def add_checkpoints_to_frame(frame, checkpoints):
    """
    Add energy checkpoint information to a frame
    """
    cps_e = [cp[0] for cp in checkpoints]
    cps_dist = [cp[1] for cp in checkpoints]
    frame['MELCheckpointsE'] = dataclasses.I3VectorDouble(cps_e)
    frame['MELCheckpointsDist'] = dataclasses.I3VectorDouble(cps_dist)

def get_checkpoints_from_frame(frame):
    """
    Get energy checkpoint information from a frame
    """
    checkpoints = []
    iterator = itertools.izip(frame['MELCheckpointsE'], frame['MELCheckpointsDist'])
    for e, dist in iterator:
        checkpoints.append((e, dist))
    return tuple(checkpoints)

def get_is_in_sim_vol(sim_vol_length=1600, sim_vol_radius=800):
    """
    Get a lambda function that determines if a muon is in the simulation volume
    """
    sim_vol_top = sim_vol_length / 2
    sim_vol_bottom = -sim_vol_top
    is_in_sim_vol = lambda m,top=sim_vol_top,bot=sim_vol_bottom,r=sim_vol_radius,x=muon_p_pos_x,y=muon_p_pos_y,z=muon_p_pos_z: (m[z] < top and m[z] > bot and (m[x]**2 + m[y]**2)**(0.5) < r)
    return is_in_sim_vol

class histogram_bundle(object):
    def __init__(self, hists=[]):
        self.hists = hists
    def accumulate(self, hb):
        for h0, h1 in itertools.izip(self.hists, hb.hists):
            h0.accumulate(h1)

class histogram_nd(object):
    """ Histogram class for creating weighted average plot """
    def __init__(self, n, bins, store_data=False, internals=None):
        if internals is not None:
            for key in internals.keys():
                setattr(self, key, internals[key])
            return

        self.n = n
        self.bins = np.array(bins)
        self.size = len(bins) - 1
        self.store_data = store_data
        size = self.size
        self.weights = np.zeros(size)
        self.weights2 = np.zeros(size)
        self.x_weighted = np.zeros(size)
        self.y_weighted = np.zeros((n, size))
        self.x2_weighted = np.zeros(size)
        self.y2_weighted = np.zeros((n, size))

        if self.store_data:
            self.x_data = []
            self.y_data = np.zeros((n, 0)).tolist()
            self.bin_data = []
            self.weights_data = []
        else:
            self.x_data = None
            self.y_data = None
            self.bin_data = None
            self.weights_data = None

    def get_internals(self):
        return self.__dict__

    def add(self, x, y, weights, bin=None):
        if self.store_data:
            if len(y) != len(self.y_data):
                raise ValueError("y length must match y_data!")
            self.x_data += list(x)
            for i in  xrange(len(y)):
                self.y_data[i] += list(y[i])
            self.weights_data += list(weights)
            if bin:
                self.bin_data += bin
            else:
                self.bin_data += ([None]*len(y))

        x = np.array(x)
        y = np.array(y)
        weights = np.array(weights)

        if bin:
            bin = np.array(bin)
            weights_by_bin = np.bincount(bin, minlength=self.size, weights=weights)
            weights2_by_bin = np.bincount(bin, minlength=self.size, weights=weights**2)
            x_weighted_by_bin = np.bincount(bin, minlength=self.size, weights=x * weights)
            x2_weighted_by_bin = np.bincount(bin, minlength=self.size, weights=x**2 * weights)
            y_weighted_by_bin = np.zeros((self.n, self.size))
            y2_weighted_by_bin = np.zeros((self.n, self.size))
            for i in xrange(len(y)):
                y_weighted_by_bin[i] = np.bincount(bin, minlength=self.size, weights=y[i] * weights)
                y2_weighted_by_bin[i] = np.bincount(bin, minlength=self.size, weights=y[i]**2 * weights)
        else:
            weights_by_bin, edges = np.histogram(x, bins=self.bins, weights=weights)
            weights2_by_bin, edges = np.histogram(x, bins=self.bins, weights=weights**2)
            x_weighted_by_bin, edges = np.histogram(x, bins=self.bins, weights=x * weights)
            x2_weighted_by_bin, edges = np.histogram(x, bins=self.bins, weights=x**2 * weights)
            y_weighted_by_bin = np.zeros((self.n, self.size))
            y2_weighted_by_bin = np.zeros((self.n, self.size))
            for i in xrange(len(y)):
                y_weighted_by_bin[i], edges = np.histogram(x, bins=self.bins, weights=y[i] * weights)
                y2_weighted_by_bin[i], edges = np.histogram(x, bins=self.bins, weights=y[i]**2 * weights)

        self.weights += weights_by_bin
        self.weights2 += weights2_by_bin
        self.x_weighted += x_weighted_by_bin
        self.y_weighted += y_weighted_by_bin
        self.x2_weighted += x2_weighted_by_bin
        self.y2_weighted += y2_weighted_by_bin

        return {'w': weights_by_bin, 'w2': weights2_by_bin, 'x': x_weighted_by_bin, 'y': y_weighted_by_bin, 'x2': x2_weighted_by_bin, 'y2': y2_weighted_by_bin}

    def accumulate(self, hist):
        if not isinstance(hist, histogram_nd):
            raise ValueError("Trying to accumulate non histogram object")
        if not hist.n == self.n:
            raise ValueError("Histogram must have same dimension")
        same_shape = (self.weights.shape == hist.weights.shape and
                self.weights2.shape == hist.weights2.shape and
                self.x_weighted.shape == hist.x_weighted.shape and
                self.y_weighted.shape == hist.y_weighted.shape and
                self.x2_weighted.shape == hist.x2_weighted.shape and
                self.y2_weighted.shape == hist.y2_weighted.shape)
        if not same_shape:
            raise ValueError("Histogram shapes are not the same")

        self.weights += hist.weights
        self.weights2 += hist.weights2
        self.x_weighted += hist.x_weighted
        self.y_weighted += hist.y_weighted
        self.x2_weighted += hist.x2_weighted
        self.y2_weighted += hist.y2_weighted

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
        w = self.get_w()
        y = self.y_weighted / ([w]*self.n)
        return y
    def get_x2(self):
        return self.x2_weighted / self.get_w()
    def get_y2(self):
        w = self.get_w()
        y2 = self.y2_weighted / ([w]*self.n)
        return y2
    def stddev(self, x, x2):
        variance = np.zeros(x.shape)
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
        return self.get_y_stddev() * ([np.sqrt(self.get_w2()) / self.get_w()]*self.n)

class histogram(histogram_nd):
    """ Histogram class for creating weighted average plot """
    def __init__(self, bins, store_data=False):
        super(histogram, self).__init__(1, bins, store_data)

    def add(self, x, y, weights, bin=None):
        return super(histogram, self).add(x, [y], weights, bin)
    def get_y(self):
        return super(histogram, self).get_y()[0]
    def get_y2(self):
        return super(histogram, self).get_y2()[0]
    def get_y_stddev(self):
        return super(histogram, self).get_y_stddev()[0]
    def get_y_stddev_of_mean(self):
        return self.get_y_stddev() * np.sqrt(self.get_w2()) / self.get_w()

def write_hist_to_file(file_handle, hist):
    internals = hist.get_internals()

class muon(object):
    def __init__(self, checkpoints, losses, weight, mu_info, nu_info, run_id, event_id):
        self.checkpoints = tuple([tuple(cp) for cp in checkpoints])
        self.losses = tuple([tuple(loss) for loss in losses])
        self.mu_info = mu_info
        self.nu_info = nu_info
        self.run_id = run_id
        self.event_id = event_id
        self.valid_checkpoints = get_valid_checkpoints(self.checkpoints)
        self.enum_vcp = [e for e in enumerate(self.valid_checkpoints)]
        self.losses = add_loss_sum(self.losses, self.valid_checkpoints)
        self.min_range, self.max_range = get_track_range(self.checkpoints)
        self.loss_infos = [get_loss_info_(cp1, cp2, self.losses) for cp1, cp2 in itertools.izip(self.valid_checkpoints[:-1], self.valid_checkpoints[1:])]

    def get_loss_info(self, x):
        bounds = get_bouning_elements(x, self.enum_vcp, lambda elem: elem[1][1])
        cp1 = bounds[0][1]
        cp2 = bounds[1][1]
        if cp1 is not None and cp2 is not None:
            loss_info = self.loss_infos[bounds[0][0]]
        else:
            loss_info = [None, None]

        return cp1,cp2,loss_info

    def get_energy(self, x, inclusive=True):
        cp1, cp2, loss_info = self.get_loss_info(x)
        return get_energy__(cp1, cp2, loss_info[0], loss_info[1], inclusive=inclusive, has_sum=True)

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
    cp1, cp2, losses, has_sum = stuff
    return get_loss_info_(cp1, cp2, losses, has_sum)

def get_loss_info_(cp1, cp2, losses, has_sum):
    """
    Get the loss rate and losses between two energy checkpoints.
    Takes single tuple argument (checkpoint1, checkpoint2, losses) to allow fast memoization.
    A checkpoint has the structure: (energy, distance along track)
    """
    #cp1, cp2, losses, has_sum = stuff
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
    """
    Get the two elements of a list that bound a value
    """
    if sort:
        l = sorted(l, key=key)
    return (next(itertools.dropwhile(lambda elem: key(elem) > x, reversed(l)), None),
            next(itertools.dropwhile(lambda elem: key(elem) < x, l), None))
def get_energy(x, checkpoints, loss_tuples, inclusive=True, has_sum=True):
    stuff = (x, checkpoints, loss_tuples, inclusive, has_sum)
    return get_energy_(stuff)

@memodict
def get_energy_(stuff):
    x, checkpoints, loss_tuples, inclusive, has_sum = stuff
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

    return energy

def get_energy__(x, cp1, cp2, loss_rate, losses, inclusive, has_sum):
    """
    Get the energy of a muon track at a point x.
    Given energy checkpoints and losses along track.
    """
    # If the checkpoints are the same then x has the same distance as the checkpoint
    if cp1 == cp2:
        return cp1[0]

    # Return 0 for regions in which we don't have enough information
    if not cp1:
        return 0
    if not cp2:
        return 0

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

    return energy

def integrate_track_energy(x1, x2, checkpoints, loss_tuples, has_sum=True):
    """
    Integrate the track energy between two points
    """
    keep_point = lambda p: (p[1] > x1) and (p[1] < x2)
    points = sorted([cp[1] for cp in checkpoints if keep_point(cp)] + [loss[1] for loss in loss_tuples if keep_point(loss)] + [x2])

    area = 0

    E1 = get_energy(x1, checkpoints, loss_tuples, inclusive=True, has_sum=has_sum)
    for x2 in points:
        E2 = get_energy(x2, checkpoints, loss_tuples, inclusive=False, has_sum=has_sum)
        area += (x2 - x1)*(E1 + E2)/2.0
        x1 = x2
        E1 = get_energy(x1, checkpoints, loss_tuples, inclusive=True, has_sum=has_sum)

    return area

def get_info_from_file(infile):
    """
    Get histogram and binning information from a pickle file
    """
    inpickle = open(infile, 'rb')
    hists = pickle.load(inpickle)
    binnings = pickle.load(inpickle)

    inpickle.close()

    return (hists, binnings)

def aggregate_info_from_dirs(dirs):
    """
    Aggregate information from multiple files
    """
    aggregated_hists = None
    aggregated_binnings = None
    for dir in dirs:
        files = glob.glob(dir + "*.pkl")
        for file in files:
            hists, binnings = get_info_from_file(file)
            if aggregated_hists is None:
                aggregated_hists = hists
                aggregated_binnings = binnings
            else:
                if (aggregated_binnings == binnings).all() and len(hists) == len(aggregated_hists):
                    for i in xrange(len(hists)):
                        aggregated_hists[i].accumulate(hists[i])
                else:
                    raise ValueError("Histograms and binnings must match!")
    return aggregated_hists

def save_info_to_file(outfile, hists, binnings):
    """
    Save histogram and binning information to a pickle file
    """
    outpickle = open(outfile, 'wb')
    pickle.dump(hists, outpickle, -1)
    pickle.dump(binnings, outpickle, -1)

    outpickle.close()

def read_from_json_file(file):
    """
    Read function for json files
    """
    return json.loads(file.readline())

def read_from_pkl_file(file):
    """
    Read function for pkl files
    """
    return pickle.load(file)

def write_to_json_file(f, obj):
    json.dump(obj, f, check_circular=False)
    f.write('\n')

def write_to_pkl_file(f, obj):
    pickle.dump(obj, f, -1)

def file_writer(file_name, q):
    if file_name.endswith('.pkl'):
        write = write_to_pkl_file
    elif file_name.endswith('.json'):
        write = write_to_json_file
    else:
        print 'No match to file extension'
        raise ValueError('No match for file extension!')

    scratch_dir = '/scratch/%s/' % os.environ['USER']
    if os.path.exists('/scratch/'):
        if not os.path.exists(scratch_dir):
            try:
                os.makedir(scratch_dir)
            except:
                pass

    is_scratch = os.path.exists(scratch_dir)
    is_scratch = False

    if is_scratch:
        print 'Using scratch'
    else:
        print 'Not using scratch'
    
    if is_scratch:
        scratch_file_name = scratch_dir + str(uuid.uuid4())
        print file_name
        print scratch_file_name
        scratch_file = open(scratch_file_name, 'w')
        print 'Opening scratch file'
        write_file = scratch_file
    else:
        write_file = open(file_name, 'w')
        print 'Opening file'

    while True:
        try:
            elems = q.get(True, 0.1)                                                                                                                                             
            if elems is Queue.Empty:
                print 'Ending writer thread'
                write_file.close()
                if is_scratch:
                    print shutil.move(scratch_file_name, file_name)
                return
            elems.reverse()
            write(write_file, elems)
        except Queue.Empty as e:
            print 'Empty'
            pass
        except Exception as e:
            print 'Got other exception'
            print e
            raise

def process_DAQ(frame, q, n, is_mono = False):
    tree = frame['I3MCTree']
    tracks = frame['MMCTrackList']
    if not is_mono:
        weight_dict = frame['I3MCWeightDict']
        header = frame['I3EventHeader']
        tree_parent = tree.parent

    primaries = tree.primaries

    tracks = [track for track in tracks if track.particle.type in mu_set] # Only tracks that are muons
    tracks = [track for track in tracks if tree.has(track.particle)] # Only tracks in MC tree

    if is_mono:
        muon_track = max(tracks, key=lambda x: x.particle.energy)
    else:
        tracks = [track for track in tracks if tree_parent(track.particle).type in nu_set] # Only tracks that have nu parent
        tracks = [track for track in tracks if tree_parent(track.particle) in primaries] # Only tracks that have primary parent
        tracks = [track for track in tracks if track.particle.energy >= 5000] # Only muons that are at least 5TeV at creation

        nu_primaries_of_tracks = np.unique([tree_parent(track.particle) for track in tracks])
        tracks_by_primary = [[track for track in tracks if tree_parent(track.particle) == p] for p in nu_primaries_of_tracks]
        max_E_muons_by_primary = [max(tracks_for_primary, key=self.get_energy) for tracks_for_primary in tracks_by_primary]
        n_muons = len(max_E_muons_by_primary)
        if(n_muons > 1):
            raise ValueError('There is more than one muon in the frame')
        if(n_muons < 1):
            return
        nu = nu_primaries_of_tracks[0]

    muon_p = muon_track.particle
    #HighEMuonTracks.append(muon_track)
    losses = [d for d in tree.get_daughters(muon_p) if d.type in loss_set] #Get the muon daughters

    # Create loss tuples
    loss_tuples = [(loss.energy, abs(loss.pos - muon_p.pos), int(loss.type)) for loss in losses]

    # Create checkpoints
    checkpoints = [(muon_p.energy, 0)]

    muon_pos_i = dataclasses.I3Position(muon_track.xi, muon_track.yi, muon_track.zi)
    checkpoints.append((muon_track.Ei, abs(muon_pos_i - muon_p.pos)))

    muon_pos_c = dataclasses.I3Position(muon_track.xc, muon_track.yc, muon_track.zc)
    checkpoints.append((muon_track.Ec, abs(muon_pos_c - muon_p.pos)))

    muon_pos_f = dataclasses.I3Position(muon_track.xf, muon_track.yf, muon_track.zf)
    checkpoints.append((muon_track.Ef, abs(muon_pos_f - muon_p.pos)))

    checkpoints.append((0, muon_p.length))

    # Calculate weights
    if is_mono:
        nu_weight = 1e-5
    else:
        nu_cos_zenith = np.cos(nu.dir.zenith)
        nu_p_int = weight_dict['TotalInteractionProbabilityWeight']
        nu_unit = I3Units.cm2/I3Units.m2
        nu_weight = nu_p_int*(flux(nu.type, nu.energy, nu_cos_zenith)/nu_unit)/generator(nu.energy, nu.type, nu_cos_zenith)

    data = []

    data.append(loss_tuples)
    data.append(nu_weight)
    data.append(checkpoints)

    data.append((muon_p.energy, muon_p.pos.x, muon_p.pos.y, muon_p.pos.z, muon_p.dir.zenith, muon_p.dir.azimuth, muon_p.length))
    if is_mono:
        data.append((0, 0, 0, 0, 0, 0, 0))
        data.append(0)
        data.append(n)
    else:
        data.append((nu.energy, nu.pos.x, nu.pos.y, nu.pos.z, nu.dir.zenith, nu.dir.azimuth, nu.length))
        data.append(header.run_id)
        data.append(header.event_id)

    q.put([data])


def file_reader(file_name, q, max_scratch_size = 1024**3):
    """
    Make a file reader based on file extension and /scratch/$USER/ availability
    """
    max_i3_processes
    is_i3 = False

    #print 'In file reader thread'
    if file_name.endswith('.pkl'):
        read = read_from_pkl_file
        print 'Got a pkl file!'
    elif file_name.endswith('.json'):
        #read = read_from_json_file
        read = lambda f: f.readline()
        print 'Got a json file!'
    elif file_name.endswith('.i3.gz') or file_name.endswith('i3.bz2'):
        is_i3 = True
    else:
        raise ValueError('No match for file extension!')

    print 'Good file extension'

    scratch_dir = '/scratch/%s/' % os.environ['USER']
    if os.path.exists('/scratch/'):
        if not os.path.exists(scratch_dir):
            try:
                os.makedir(scratch_dir)
            except:
                pass

    is_scratch = os.path.exists(scratch_dir)
    is_scratch = is_scratch and os.stat(file_name).st_size <= max_scratch_size

    print 'Is scratch: %d' % is_scratch

    if is_scratch:
        scratch_file_name = scratch_dir + str(uuid.uuid4())
        print file_name
        print scratch_file_name
        print shutil.copy(file_name, scratch_file_name)
        read_file_name = scratch_file_name
    else:
        print file_name
        read_file_name = file_name

    if is_i3:
        is_mono = 'mono' in file_name
        pool = multiprocessing.Pool(2, maxtasksperchild=100)
        class MCMuonInfoTray(icetray.I3ConditionalModule):
            def __init__(self, context):
                super(MyModule, self).__init__(context)
                self.n = None
            def Configure(self):
                pass
            def Physics(self, frame):
                pass
            def DAQ(self, frame):
                if self.n == None:
                    self.n = 0
                else:
                    self.n += 1
                result = pool.apply_async(process_DAQ, (frame, q, self.n, is_mono))
        tray = I3Tray()
        tray.Add("I3Reader", "my_reader", FilenameList=[read_file_name])
        tray.Add(lambda frame: frame.Has('I3MCTree'))
        tray.Add(lambda frame: frame.Has('MMCTrackList'))
        tray.Add(MCMuonInfoTray)
        tray.Execute()
        tray.Finish()
        pool.close()
        pool.join()
        q.put('')
    else:
        stop = False
        elems = []
        while True:
            try:
                elems = read(read_file)
                if elems == '':
                    q.put(elems)
                    raise ValueError('Nothing left in file')
                q.put(elems)
                print 'Put'
            except Exception as e:
                print 'Got exception: %s' % str(e)
                print 'Issue reading more from file'
                print 'Ending reader thread'
                if is_scratch:
                    os.remove(scratch_file_name)
                return