# Set up the argument parser
import argparse
import glob
import os.path as path
import file_utils

if __name__ == '__main__':
    default_geo = '/data/sim/sim-new/downloads/GCD_09_07_12/GeoCalibDetectorStatus_IC86.55697_corrected_V2.i3.gz'
    default_infiles = '/data/ana/IC79/numu_forward_folding/sim/Alfa/Lnu/IC86/eff0.9900/1.i3.gz'

    parser = argparse.ArgumentParser(description='Proccess filenames')
    parser.add_argument('-i', '--infiles', metavar='infiles', type=str, default=default_infiles)
    parser.add_argument('-g', '--geo', metavar='geo', type=str, default=default_geo)
    parser.add_argument('-o', '--outdir', default='')
    parser.add_argument('-f', '--histogram-file', dest='hist_file', default='./histogram_output')

    parser.add_argument('-r', '--range', default='')
    parser.add_argument('-s', '--sample', type=float, default=1.0)
    parser.add_argument('-d', '--isdata', type=float, default=1.0)

    args = parser.parse_args()
    infile_string = args.infiles.strip('"\'')
    infiles = glob.glob(infile_string)
    file_utils.sort_nicely(infiles)
    geo_string = args.geo.strip('"\'')
    geo = glob.glob(geo_string)[0]
    outdir = args.outdir.strip('"\'')
    hist_file = args.hist_file.strip('"\'')
    file_range = args.range.translate(None, '\\/:!@#$%Z^&*()+=`;"\'?><,|{}_ ') # Remove anything weird before storing
    if file_range == '':
        file_range = None
    else:
        file_range = [int(i) for i in file_range.split('-', 1)] # Split into 1 or 2 numbers
    sampling_factor = (lambda x: min(x, 1.0/x))(args.sample)
    is_data = args.isdata

    print args
    print 'all infiles: %s' % str(infiles)

    file_pairs = file_utils.get_output_files(infiles, outdir, mkdir=False, file_range=file_range)
    print file_pairs

# Millipede imports
from I3Tray import *
import sys
import icecube
from icecube import icetray, dataio, dataclasses, simclasses, photonics_service, VHESelfVeto, phys_services, NewNuFlux
from icecube.icetray import I3Units
import icecube.weighting.weighting as weighting
from icecube.weighting.weighting import from_simprod
load('millipede')

import numpy as np
import pickle
import histogram

_pulse_series = 'TTPulses'
_reco_track = 'MPEFit_TT'

# Initialize the flux and generator for the sample
# Parameters below are for /data/ana/IC79/numu_forward_folding/sim/Alfa/Lnu/IC86/eff0.9900/
_flux_name = 'honda2006'
_flux = NewNuFlux.makeFlux(_flux_name).getFlux
_generator = weighting.NeutrinoGenerator(1e5, 200, 1e9, 2, 'NuMu',
    InjectionMode = 'Surface',
    ZenithMin = 80*I3Units.deg,
    ZenithMax = 180*I3Units.deg,
    AzimuthMin = 0*I3Units.deg,
    AzimuthMax = 360*I3Units.deg,
    CylinderRadius = 800*I3Units.meter,
    CylinderHeight = 1000.*I3Units.m
)

# Initialize photon table services
table_base = os.path.expandvars('$I3_DATA/photon-tables/splines/emu_%s.fits')
_muon_service = photonics_service.I3PhotoSplineService(table_base % 'abs', table_base % 'prob', 0)
table_base = os.path.expandvars('$I3_DATA/photon-tables/splines/ems_spice1_z20_a10.%s.fits')
_cascade_service = photonics_service.I3PhotoSplineService(table_base % 'abs', table_base % 'prob', 0)
    
def init(pulse_series=None, reco_track=None, flux=None, generator=None, muon_service=None, cascade_service=None):
    if pulse_series is not None:
        global _pulse_series
        _pulse_series = pulse_series
    if reco_track is not None:
        global _reco_track
        _reco_track = reco_track
    if flux is not None:
        global _flux
        _flux = flux
    if generator is not None:
        global _generator
        _generator = generator
    if muon_service is not None:
        global _muon_service
        _muon_service = muon_service
    if cascade_service is not None:
        global _cascade_service
        _cascade_service = cascade_service

# Ensures that a module only sees P frames
def phys(f):
    def ff(frame):
        if frame.Stop == icetray.I3Frame.Physics:
            return f(frame)
        else:
            return True
    return ff

# Module that counts the number of frames seen
counters = []
def counter(func, const=False):    
    global counters
    i = len(counters)
    counters.append(0)
    def f(frame):
        res = func(frame)
        counters[i] += int(res)
        if const:
            return True
        else:
            return res 
    return f

class muon_energy_info:
    def __init__(self, muon_track, muon_p, muon_losses):
        # Create loss tuples
        self.losses = sorted([[loss.energy, abs(loss.pos - muon_p.pos), int(loss.type)] for loss in muon_losses], key=lambda x: x[1])

        # Create checkpoints
        self.checkpoints = [(muon_p.energy, 0)]

        muon_pos_i = dataclasses.I3Position(muon_track.xi, muon_track.yi, muon_track.zi)
        self.checkpoints.append((muon_track.Ei, abs(muon_pos_i - muon_p.pos)))

        muon_pos_c = dataclasses.I3Position(muon_track.xc, muon_track.yc, muon_track.zc)
        self.checkpoints.append((muon_track.Ec, abs(muon_pos_c - muon_p.pos)))

        muon_pos_f = dataclasses.I3Position(muon_track.xf, muon_track.yf, muon_track.zf)
        self.checkpoints.append((muon_track.Ef, abs(muon_pos_f - muon_p.pos)))

        self.checkpoints.append((0, muon_p.length))

        # Assign valid checkpoints
        track_cps = self.checkpoints[1:-1]
        self.valid_checkpoints = [self.checkpoints[0]] + [cp for cp in track_cps if cp[0] > 0] + [self.checkpoints[-1]]
        self.valid_checkpoints = sorted(self.valid_checkpoints, key=lambda x: x[1])
        
        # Add loss sums to losses
        next_dist = 0
        total = 0
        for j in xrange(len(self.losses)):
            if self.losses[j][1] >= next_dist:
                next_dist = next(itertools.dropwhile(lambda cp: cp[1] <= self.losses[j][1], self.valid_checkpoints), (None, np.inf))[1]
                total = 0
            total += self.losses[j][0]
            self.losses[j] = tuple(self.losses[j] + [total])

        self.loss_rates = []
        self.loss_ranges = []
        for i in xrange(0, len(self.valid_checkpoints)-1):
            cp1 = self.valid_checkpoints[i]
            cp2 = self.valid_checkpoints[i+1]
            first_index = next(itertools.dropwhile(lambda l: l[1][1] <= cp1[1], enumerate(self.losses)))[0]
            last_index = len(self.losses) - 1 - next(itertools.dropwhile(lambda l: l[1][1] >= cp2[1], enumerate(reversed(self.losses))), [0])[0]

            if last_index < 0:
                total_stochastic_loss = 0
            else:
                total_stochastic_loss = self.losses[last_index][3]
            loss_rate = (cp1[0] - cp2[0] - total_stochastic_loss) / (cp2[1] - cp1[1])
            self.loss_rates.append(loss_rate)
            self.loss_ranges.append((first_index, last_index+1))

    def get_energy(self, x):
        """  
        Get the energy of a muon track at a point x.
        Given energy checkpoints and losses along track.
        """
        # Get the checkpoints on either side of x, search by distance
        cp2_i = next(itertools.dropwhile(lambda elem: elem[1][1] < x, self.valid_checkpoints), [-1])[0]

        # Return muon energy before track begins, return 0 beyond track
        if cp2_i == 0:
            return self.checkpoints[0][0]
        if cp2_i < 0:
            return 0

        cp1_i = cp2_i - 1
        cp1 = self.valid_checkpoints(cp1_i)
        cp2 = self.valid_checkpoints(cp2_i)

        if x == cp1[1]:
            return cp1[0]
        if x == cp2[1]:
            return cp2[0]

        # Get the loss rate and losses between the checkpoints
        loss_rate = self.loss_rates[cp1_i]
        losses_begin, losses_end = self.loss_ranges[cp1_i]

        # Get the sum of losses between x and the checkpoint before x
        stoch_loss_since_cp1 = 0
        if losses_begin != losses_end:
            i_loss_before_x = next(itertools.dropwhile(lambda loss: loss[1][1] <= x, enumerate(self.losses[losses_begin:losses_end])), [losses_end-losses_begin])[0] - 1
            if i_loss_before_x >= 0:
                stoch_loss_since_cp1 = self.losses[loss_begin+i_loss_before_x][3]

        # (E at last cp) - (stoch losses since last cp) - (loss rate * distance from last cp)
        energy = cp1[0] - stoch_loss_since_cp1 - (x - cp1[1]) * loss_rate

        return energy

# Stores the result of a function in an I3Map
# Useful for storing results of different cuts
def store_in_map(func, label=None, map_name='CutMap'):
    if label is None:
        label = func.__name__
    @phys
    def f(frame):
        res = func(frame)
        frame[map_name][label] = res
    return f

# Create signatures for the combinations of cuts that a frame passes
def get_cut_map_signatures(frame, cut_maps):
    sigs = [[]]
    cut_maps = sorted(cut_maps)
    for cut_map in cut_maps:
        new_sigs = []
        for sig in sigs:
            keys = sorted(frame[cut_map].keys())
            for k in keys:
                if frame[cut_map][k]:
                    new_sigs.append(sig + [k])
        sigs = new_sigs
    sigs = [tuple(sig) for sig in sigs]
    return sigs

# Gets the weight of the neutrino event
def get_weight(frame):
    tree = frame['I3MCTree']
    nu = tree.get_primary(frame['TrueMuonTrack'])
    weight_dict = frame['I3MCWeightDict']
    nu_cos_zenith = np.cos(nu.dir.zenith)
    nu_p_int = weight_dict['TotalInteractionProbabilityWeight']
    nu_unit = I3Units.cm2/I3Units.m2
    gen_w = _generator(nu.energy, nu.type, nu_cos_zenith)
    if gen_w <= 0:
        print
        print
        print 'I3EventHeader', frame['I3EventHeader']
        print 'I3MCTreeGen', frame['I3MCTreeGen']
        print 'I3MCWeightDict', frame['I3MCWeightDict']
        print 'TrueMuonTrack', frame['TrueMuonTrack']
        print 'TrueNuTrack', frame['TrueNuTrack']
        print 'MMCTrackList'
        for t in frame['MMCTrackList']:
            print t.particle
        return None
    nu_weight = nu_p_int*(_flux(nu.type, nu.energy, nu_cos_zenith)/nu_unit)/gen_w
    return nu_weight

# Create a collection of histograms for track information
def make_track_collection():
    collection = dict()
    collection['rcap'] = histogram.histogram(bins=np.linspace(0, 3000, 3000/10+1))
    collection['zcap'] = histogram.histogram(bins=np.linspace(-3000, 3000, 6000/10+1))
    
    collection['startdelta'] = histogram.histogram(bins=np.linspace(0, 3000, 3000/10+1))
    collection['centerdelta'] = histogram.histogram(bins=np.linspace(0, 3000, 3000/10+1))
    
    collection['consideredlength'] = histogram.histogram(bins=np.linspace(0, 3000, 3000/10+1))
    collection['geolength'] = histogram.histogram(bins=np.linspace(0, 3000, 3000/10+1))
    collection['cherenkovlength'] = histogram.histogram(bins=np.linspace(0, 3000, 3000/10+1))

    return collection

# Compute and add track information to a set of histogram collections
def track_histogram_module(collections, pulse_series=_pulse_series, track=_reco_track, length=600, cut_maps=[]):
    @phys
    def f(frame):
        if track  not in frame.keys():
            return
        sigs = get_cut_map_signatures(frame, cut_maps)
        keys = collections.keys()
        for sig in sigs:
            if sig not in keys:
                collections[sig] = make_track_collection()

        weight = get_weight(frame)

        if weight is None:
            print 'track_histogram_module'
            print track, frame[track]
            print
            print
            return

        def add(l, x, y=[1]):
            for sig in sigs:
                collections[sig][l].add(x,y,[weight])

        def get_from_frame(x):
            if x in frame.keys():
                x = frame[x]
            else:
                x = None
            return x

        origin_cap = phys_services.I3Calculator.closest_approach_position(frame[track],dataclasses.I3Position(0,0,0))
        add('rcap', [origin_cap.r])
        add('zcap', [origin_cap.z])

        track_start = 'TrackStart' + pulse_series + track
        track_geo_start = 'TrackGeoBoundsStart' + track
        track_geo_end = 'TrackGeoBoundsEnd' + track
        track_c_start = 'TrackBoundsStart' + pulse_series + track
        track_c_end = 'TrackBoundsEnd' + pulse_series + track

        track_start = get_from_frame(track_start)
        track_geo_start = get_from_frame(track_geo_start)
        track_geo_end = get_from_frame(track_geo_end)
        track_c_start = get_from_frame(track_c_start)
        track_c_end = get_from_frame(track_c_end)
        
        if track_start is not None:
            if track_geo_start is not None:
                start_delta = abs(track_start - track_geo_start)
                add('startdelta', [start_delta])
                add('centerdelta', [start_delta + (abs(track_geo_end - track_start) - length) / 2.0])
            if track_geo_end is not None:
                add('consideredlength', [abs(track_geo_end - track_start)])
                if track_geo_start is not None:
                    add('geolength', [abs(track_geo_end - track_geo_start)])
        if track_c_start is not None and track_c_end is not None:
            add('cherenkovlength', [abs(track_c_end - track_c_start)])
    return f

# Create a collection of histograms for energy information
def make_energy_collection():
    collection = dict()
    min_E, max_E = -2, 5
    bins_per_decade = 10
    collection['deltaE'] = histogram.histogram(bins=np.logspace(min_E,max_E,(max_E-min_E)*bins_per_decade+1))
    collection['frac_loss'] = histogram.histogram_nd(2, bins=np.linspace(0,600,600/60+1))
    return collection

# Compute and add energy information to a set of histogram collections
def energy_histogram_module(collections, pulse_series, track, cut_maps=[], length=600, center=False, losses=None):
    millipede_key = 'MillipedeHighEnergy'+pulse_series+track
    track_start_key = 'TrackStart'+pulse_series+track
    track_end_key = 'TrackGeoBoundsEnd'+track
    if losses is not None:
        millipede_key = losses
        track_start_key = 'TrackStart'+losses
    @phys
    def f(frame):
        if track not in frame.keys() or millipede_key not in frame.keys() or track_start_key not in frame.keys():
            return
        if frame[track_start_key] is None or frame[track_end_key] is None:
            return
        sigs = get_cut_map_signatures(frame, cut_maps)
        keys = collections.keys()
        for sig in sigs:
            if sig not in keys:
                collections[sig] = make_energy_collection()

        weight = get_weight(frame)

        if weight is None:
            print 'energy_histogram_module'
            print track, frame[track]
            print 
            print 
            return

        def add(l, x, y=[1], w=[weight]):
            for sig in sigs:
                collections[sig][l].add(x,y,w)
        
        frame_track = frame[track]
        track_start = frame[track_start_key]
        track_end = frame[track_end_key]
        losses = [p for p in frame[millipede_key] if p.energy > 0]
        considered_track_start = track_start
        if center:
            start = (abs(track_end - track_start) - length) / 2.0 
            end = start + length
            considered_track_start = track_start + frame_track.dir*start
        losses = [p for p in losses if (lambda x: x >= 0 and x <= length)((p.pos - considered_track_start)*frame_track.dir)]
        deltaE = sum([p.energy for p in losses])
        
        add('deltaE', [deltaE])
        if(deltaE == 0): 
            return

        bins = np.linspace(0,length,length/60+1)
        point_frac_dE = []
        point_dE = []
        point_x = []
        for i in xrange(len(bins)-1):
            bin_dE = sum([p.energy for p in losses if (lambda x: x >= bins[i] and x < bins[i+1])((p.pos - considered_track_start)*frame_track.dir)])
            point_dE.append(bin_dE)
            point_frac_dE.append(bin_dE / deltaE)
            point_x.append(bins[i])
        add('frac_loss', point_x, [point_frac_dE, point_dE], w=[weight]*len(point_frac_dE))
    return f

# Compute and add energy information to a set of histogram collections
def true_energy_histogram_module(collections, track='TrueMuonTrack', cut_maps=[], track_start_key=None, track_end_key=None, length=600, center=False, losses='TrueMuonTrack'):
    if track_end_key is None:
        track_end_key = 'TrackGeoBoundsEnd'+track
    if track_start_key is None:
        track_start_key = 'TrackStart' + losses
    @phys
    def f(frame):
        if track not in frame.keys() or losses not in frame.keys():
            return False
        if frame[track_start_key] is None or frame[track_end_key] is None:
            return False
        sigs = get_cut_map_signatures(frame, cut_maps)
        keys = collections.keys()
        for sig in sigs:
            if sig not in keys:
                collections[sig] = make_energy_collection()

        weight = get_weight(frame)

        if weight is None:
            print 'energy_histogram_module'
            print track, frame[track]
            print 
            print 
            return

        def add(l, x, y=[1], w=[weight]):
            for sig in sigs:
                collections[sig][l].add(x,y,w)
                    
        muon_p = frame[track]
        muon_track = [t for t in frame['MMCTrackList'] if t.particle.id == muon_p.id][0]
        muon_losses = frame[losses]

        track_start = frame[track_start_key]
        track_end = frame[track_end_key]

        track_start_d = (track_start - muon_p.pos)*muon_p.dir
        track_end_d = (track_end - muon_p.pos)*muon_p.dir

        if center:
            start = track_start_d + (abs(track_end_d - track_start_d) - length) / 2.0
        else:
            start = track_start_d
        end = start + length

        me_info = muon_energy_info(muon_track, muon_p, muon_losses)

        deltaE = me_info.get_energy(start) - me_info.get_energy(end)
            
        add('deltaE', [deltaE])
        if(deltaE == 0): 
            return

        dx = 60
        bins = np.linspace(0,length,length/dx+1)
        point_frac_dE = []
        point_dE = []
        point_x = []
        e0 = me_info.get_energy(start)
        for i in xrange(len(bins)-1):
            e1 = me_info.get_energy(start + bins[i+1])
            bin_dE = e0 - e1
            point_dE.append(bin_dE)
            point_frac_dE.append(bin_dE / deltaE)
            point_x.append(bins[i])
            e0 = e1
        add('frac_loss', point_x, [point_frac_dE, point_dE], w=[weight]*len(point_frac_dE))
    return f

# Instantiate useful sets of particle types
p = dataclasses.I3Particle()
loss_set = set([p.PairProd, p.DeltaE, p.Brems, p.NuclInt])
nu_set = set([p.Nu, p.NuE, p.NuEBar, p.NuMu, p.NuMuBar, p.NuTau, p.NuTauBar])
mu_set = set([p.MuPlus, p.MuMinus])

# Assign MC Truth information to frame objects
def set_mc_truth_track(frame):
    tree = frame['I3MCTree']
    tracks = frame['MMCTrackList']
    primaries = tree.primaries

    tree_parent = tree.parent

    tracks = [track for track in tracks if track.particle.type in mu_set] # Only tracks that are muons
    tracks = [track for track in tracks if tree.has(track.particle)] # Only tracks in MC tree
    #tracks = [track for track in tracks if tree_parent(track.particle) in primaries] # Only tracks that have primary parent
    nu_primaries_of_tracks = np.unique([tree_parent(track.particle) for track in tracks])
    tracks_by_primary = [[track for track in tracks if tree_parent(track.particle) == p] for p in nu_primaries_of_tracks]
    max_E_muons_by_primary = [max(tracks_for_primary, key=lambda x: x.particle.energy) for tracks_for_primary in tracks_by_primary]
    max_E_muon_i = max(enumerate(max_E_muons_by_primary), key=lambda x: x[1].particle.energy)[0]
    n_muons = len(max_E_muons_by_primary)
    #if(n_muons > 1):
    #    raise ValueError('There is more than one muon in the frame')
    if(n_muons < 1):
        print frame['I3EventHeader'].event_id
        return
    nu = nu_primaries_of_tracks[max_E_muon_i]
    muon_track = max_E_muons_by_primary[max_E_muon_i]

    true_losses = dataclasses.I3VectorI3Particle(tree.get_daughters(muon_track.particle))

    frame['TrueMuonLosses'] = true_losses
    frame['TrueMuonTrack'] = muon_track.particle

# Module to shift the position and time of a track to something that looks reasonable to millipede
def pre_milli_time_shift_module(track):
    def f(frame):
        if track not in frame.keys():
            return
        basep = frame[track]
        #Shift position to closest approach position to 0,0,0. Adjust time appropriately. 
        origin_cap = phys_services.I3Calculator.closest_approach_position(basep,dataclasses.I3Position(0,0,0))
        basep_shift_d = (origin_cap - basep.pos) * basep.dir
        basep_shift_t = basep_shift_d/basep.speed
        basep.pos = origin_cap
        basep.time = basep.time + basep_shift_t

        del frame[track]
        frame[track] = basep
    return f

# Module to cut on the number of hits in icecube doms
# Default mask: no deepcore hits included
def n_hits_cut_module(n_hits=25, pulse_series=_pulse_series, om_mask=(lambda om_series_pair: om_series_pair[0].string <= 78)):
    def f(frame):
        if pulse_series not in frame.keys():
            return
        frame_pulse_series = frame[pulse_series]
        if type(frame_pulse_series) == icecube.dataclasses.I3RecoPulseSeriesMapMask:
            frame_pulse_series = frame_pulse_series.apply(frame)
        assert type(frame_pulse_series) == icecube.dataclasses.I3RecoPulseSeriesMap

        number_of_hits = len([p for p in frame_pulse_series if om_mask(p)])

        return number_of_hits >= n_hits
    return f

# Module to store the geometric track bounds
def geometric_length_cut_module(geo='I3Geometry', track=_reco_track, length=600):
    @phys
    def f(frame):
        if track not in frame.keys():
            return False
        global passed_length_cut
        frame_track = frame[track]
        points = VHESelfVeto.IntersectionsWithInstrumentedVolume(frame[geo], frame_track)
        if len(points) < 2:
            return False
        pos_key = lambda pos: (pos - frame_track.pos)*frame_track.dir
        frame['TrackGeoBoundsStart'+track] = min(points, key=pos_key)
        frame['TrackGeoBoundsEnd'+track] = max(points, key=pos_key)
        return abs(points[0] - points[1]) >= length
    return f

# Module to add the time window expected by millipede
def time_window_module(pulse_series=_pulse_series):
    @phys
    def add_time_window(frame):
        frame_key = pulse_series + 'TimeRange'
        if frame_key not in frame:
            window = dataclasses.I3TimeWindow(*(lambda x: (min(x)-1000, max(x)+1000))([p.time for l in frame[pulse_series].apply(frame).values() for p in l]))
            frame[pulse_series + 'TimeRange'] = window
        return True
    return add_time_window

# Module to add a map to the frame
def map_module(name='CutMap', type=dataclasses.I3MapStringBool):
    @phys
    def f(frame):
        frame[name] = type()
    return f

# Module that accepts a frame if any cut in that level has passed
def map_cut_module(map_name):
    @phys
    def f(frame):
        return np.any(frame[map_name].values())
    return f

# Adds the time window and millipede modules to the tray
def add_millipede_module(tray, pulse_series=_pulse_series, seed_track=_reco_track):
    tray.Add(time_window_module(pulse_series=_pulse_series))
    millipede_key = 'MillipedeHighEnergy'+pulse_series+seed_track
    tray.Add('MuMillipede', 'millipede_highenergy_'+pulse_series+'_'+seed_track,
            MuonPhotonicsService=_muon_service, CascadePhotonicsService=_cascade_service,
            PhotonsPerBin=15, MuonRegularization=0, ShowerRegularization=0,
            MuonSpacing=0, ShowerSpacing=10, SeedTrack=seed_track,
            Output='MillipedeHighEnergy'+pulse_series+seed_track, Pulses=pulse_series)

# Gets the track bounds based on the known points of Cherenkov emmision
def get_track_bounds(frame, pulse_series, track):
    omgeo = frame['I3Geometry'].omgeo
    frame_track = frame[track]
    frame_pulse_series = frame[pulse_series]
    if type(frame_pulse_series) == icecube.dataclasses.I3RecoPulseSeriesMapMask:
        frame_pulse_series = frame_pulse_series.apply(frame)
    assert type(frame_pulse_series) == icecube.dataclasses.I3RecoPulseSeriesMap
    pos_candidates = []
    for dom, hits in frame_pulse_series.items():
        for hit in hits:
            time = phys_services.I3Calculator.time_residual(frame[track], omgeo[dom].position, hit.time)
            pos = phys_services.I3Calculator.cherenkov_position(frame[track], omgeo[dom].position)
            if time >= -15. and time <= 75.:
                pos_candidates.append(pos)
    pos_key = lambda pos: (pos - frame_track.pos)*frame_track.dir
    if len(pos_candidates) == 0:
        return (None, None)
    return (min(pos_candidates, key=pos_key), max(pos_candidates, key=pos_key))

# Define the bounds of the track by Cherenkov position
def track_bounds_module(pulse_series=_pulse_series, track=_reco_track):
    @phys
    def f(frame):
        if track not in frame.keys():
            return
        bounds = get_track_bounds(frame, pulse_series, track)
        frame['TrackBoundsStart'+pulse_series+track] = bounds[0]
        frame['TrackBoundsEnd'+pulse_series+track] = bounds[1]
    return f

# Define the position we want to consider as the start of the track
# Goes 100m from the first reconstructed loss, goes to the edge of the detector if this point is outside
def track_start_module(pulse_series=_pulse_series, track=_reco_track, losses=None):
    millipede_key = 'MillipedeHighEnergy'+pulse_series+track
    track_start_key = 'TrackStart'+pulse_series+track
    if losses is not None:
        millipede_key = losses
        track_start_key = 'TrackStart'+losses
    @phys
    def f(frame):
        if track not in frame.keys() or millipede_key not in frame.keys():
            return
        frame_track = frame[track]
        losses = [p for p in frame[millipede_key] if p.energy > 0]
        if len(losses) == 0:
            return
        pos_key = lambda pos: (pos - frame_track.pos)*frame_track.dir
        pos = min([p.pos for p in losses], key=pos_key) + frame_track.dir*100
        bounds = VHESelfVeto.IntersectionsWithInstrumentedVolume(frame['I3Geometry'], frame_track)
        if len(bounds) < 2:
            frame[track_start_key] = None
            return
        is_in_detector = (pos - bounds[0])*(pos - bounds[1]) < 0
        if not is_in_detector:
            pos = min(bounds, key=pos_key)
        frame[track_start_key] = pos
    return f

# Cut on the considered track length
# We consider 100m from the first reconstructed loss to the geometric exit point
def track_length_cut_module(pulse_series=_pulse_series, track=_reco_track, length=600., losses=None):
    millipede_key = 'MillipedeHighEnergy'+pulse_series+track
    track_start_key = 'TrackStart'+pulse_series+track
    track_end_key = 'TrackGeoBoundsEnd'+track
    if losses is not None:
        millipede_key = losses
        track_start_key = 'TrackStart' + losses
    @phys
    def f(frame):
        if track not in frame.keys() or millipede_key not in frame.keys():
            return False
        if frame[track_start_key] is None or frame[track_end_key] is None:
            return False
        return abs(frame[track_end_key] - frame[track_start_key]) >= length
    return f

# Cut on the reconstructed DeltaE in some length of the track
def deltaE_cut_module(pulse_series=_pulse_series, track=_reco_track, dE=500, length=600, center=False, losses=None):
    millipede_key = 'MillipedeHighEnergy'+pulse_series+track
    track_start_key = 'TrackStart'+pulse_series+track
    track_end_key = 'TrackGeoBoundsEnd'+track
    if losses is not None:
        millipede_key = losses
        track_start_key = 'TrackStart' + losses
    @phys
    def f(frame):
        if track not in frame.keys() or millipede_key not in frame.keys():
            return False
        if frame[track_start_key] is None or frame[track_end_key] is None:
            return False
        frame_track = frame[track]
        track_start = frame[track_start_key]
        track_end = frame[track_end_key]
        losses = [p for p in frame[millipede_key] if p.energy > 0]
        if center:
            start = (abs(track_end - track_start) - length) / 2.0
            end = start + length
            losses = [p for p in losses if (lambda x: x >= start and x <= end)((p.pos - track_start)*frame_track.dir)]
        else:
            losses = [p for p in losses if (lambda x: x >= 0 and x <= length)((p.pos - track_start)*frame_track.dir)]
        deltaE = sum([p.energy for p in losses])
        return deltaE >= dE
    return f

def true_deltaE_cut_module(track='TrueMuonTrack', track_start_key=None, track_end_key=None,dE=500, length=600, center=False, losses='TrueMuonTrack'):
    if track_end_key is None:
        track_end_key = 'TrackGeoBoundsEnd'+track
    if track_start_key is None:
        track_start_key = 'TrackStart' + losses
    @phys
    def f(frame):
        if track not in frame.keys() or losses not in frame.keys():
            return False
        if frame[track_start_key] is None or frame[track_end_key] is None:
            return False
        muon_p = frame[track]
        muon_track = [t for t in frame['MMCTrackList'] if t.particle.id == muon_p.id][0]
        muon_losses = frame[losses]

        track_start = frame[track_start_key]
        track_end = frame[track_end_key]

        track_start_d = (track_start - muon_p.pos)*muon_p.dir
        track_end_d = (track_end - muon_p.pos)*muon_p.dir

        if center:
            start = track_start_d + (abs(track_end_d - track_start_d) - length) / 2.0
        else:
            start = track_start_d
        end = start + length

        me_info = muon_energy_info(muon_track, muon_p, muon_losses)

        deltaE = me_info.get_energy(start) - me_info.get_energy(end)
        return deltaE >= dE
    return f
    

def run_tray(geo, infiles, outfile, hist_outfile, pulse_series=_pulse_series, reco_track=_reco_track):
    print 'Creating tray for: ', (infiles, outfile)
    
    # Instantiate the tray
    tray = I3Tray()
    tray.Add("I3Reader", "my_reader", FilenameList=[geo]+infiles)
    #tray.Add(lambda frame: random.random() <= sampling_factor)

    # Ensure everything is there
    tray.Add(phys(counter(lambda frame: frame.Has('I3Geometry'))))
    tray.Add(phys(counter(lambda frame: frame.Has('I3MCTree'))))
    tray.Add(phys(counter(lambda frame: frame.Has('MMCTrackList'))))
    tray.Add(phys(counter(lambda frame: frame.Has(reco_track))))
    tray.Add(phys(counter(lambda frame: frame.Has(pulse_series))))

    # N hits cut
    tray.Add(counter(n_hits_cut_module(pulse_series=pulse_series, n_hits=25)))

    # Add the MC Truth Track
    tray.Add(phys(set_mc_truth_track))
    tray.Add(phys(pre_milli_time_shift_module('TrueMuonTrack')))

    # Geometric length cuts
    tray.Add(map_module('GeoLengthCutMap', dataclasses.I3MapStringBool))
    tray.Add(store_in_map(geometric_length_cut_module(track=reco_track), label=reco_track+'GeoLength', map_name='GeoLengthCutMap'))
    tray.Add(store_in_map(geometric_length_cut_module(track='TrueMuonTrack'), label='TrueGeoLength', map_name='GeoLengthCutMap'))
    tray.Add(counter(map_cut_module('GeoLengthCutMap')))

    # Track Cherenkov bounds
    tray.Add(track_bounds_module(pulse_series=pulse_series, track=reco_track))
    tray.Add(track_bounds_module(pulse_series=pulse_series, track='TrueMuonTrack'))

    # Millipede
    add_millipede_module(tray, pulse_series=pulse_series, seed_track=reco_track)
    add_millipede_module(tray, pulse_series=pulse_series, seed_track='TrueMuonTrack')

    # Track start
    tray.Add(track_start_module(pulse_series=pulse_series, track=reco_track))
    tray.Add(track_start_module(pulse_series=pulse_series, track='TrueMuonTrack'))
    tray.Add(track_start_module(pulse_series=pulse_series, track='TrueMuonTrack', losses='TrueMuonLosses'))

    # Track length cuts
    tray.Add(map_module('LengthCutMap', dataclasses.I3MapStringBool))
    tray.Add(store_in_map(track_length_cut_module(pulse_series=pulse_series, track=reco_track), label=reco_track+'Length', map_name='LengthCutMap'))
    tray.Add(store_in_map(track_length_cut_module(pulse_series=pulse_series, track='TrueMuonTrack'), label='TrueLength', map_name='LengthCutMap'))
    tray.Add(counter(map_cut_module('LengthCutMap')))

    # Compute track histograms before energy cuts
    reco_track_collections = dict()
    tray.Add(track_histogram_module(reco_track_collections, pulse_series=pulse_series, track=reco_track, cut_maps=['GeoLengthCutMap', 'LengthCutMap']))
    true_track_collections = dict()
    tray.Add(track_histogram_module(true_track_collections, pulse_series=pulse_series, track='TrueMuonTrack', cut_maps=['GeoLengthCutMap', 'LengthCutMap']))

    # Delta E cuts
    tray.Add(map_module('DeltaECutMap', dataclasses.I3MapStringBool))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track=reco_track, dE=300, center=False), label=reco_track+'StartDeltaE300', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track=reco_track, dE=500, center=False), label=reco_track+'StartDeltaE500', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track=reco_track, dE=700, center=False), label=reco_track+'StartDeltaE700', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track=reco_track, dE=300, center=True), label=reco_track+'CenterDeltaE300', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track=reco_track, dE=500, center=True), label=reco_track+'CenterDeltaE500', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track=reco_track, dE=700, center=True), label=reco_track+'CenterDeltaE700', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track='TrueMuonTrack', dE=300, center=False), label='MCTMilliStartDeltaE300', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track='TrueMuonTrack', dE=500, center=False), label='MCTMilliStartDeltaE500', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track='TrueMuonTrack', dE=700, center=False), label='MCTMilliStartDeltaE700', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track='TrueMuonTrack', dE=300, center=True), label='MCTMilliCenterDeltaE300', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track='TrueMuonTrack', dE=500, center=True), label='MCTMilliCenterDeltaE500', map_name='DeltaECutMap'))
    tray.Add(store_in_map(deltaE_cut_module(pulse_series=pulse_series, track='TrueMuonTrack', dE=700, center=True), label='MCTMilliCenterDeltaE700', map_name='DeltaECutMap'))
    tray.Add(store_in_map(true_deltaE_cut_module(track='TrueMuonTrack', dE=300, center=False, losses='TrueMuonLosses'), label='MCTStartDeltaE300', map_name='DeltaECutMap'))
    tray.Add(store_in_map(true_deltaE_cut_module(track='TrueMuonTrack', dE=500, center=False, losses='TrueMuonLosses'), label='MCTStartDeltaE500', map_name='DeltaECutMap'))
    tray.Add(store_in_map(true_deltaE_cut_module(track='TrueMuonTrack', dE=700, center=False, losses='TrueMuonLosses'), label='MCTStartDeltaE700', map_name='DeltaECutMap'))
    tray.Add(store_in_map(true_deltaE_cut_module(track='TrueMuonTrack', dE=300, center=True, losses='TrueMuonLosses'), label='MCTCenterDeltaE300', map_name='DeltaECutMap'))
    tray.Add(store_in_map(true_deltaE_cut_module(track='TrueMuonTrack', dE=500, center=True, losses='TrueMuonLosses'), label='MCTCenterDeltaE500', map_name='DeltaECutMap'))
    tray.Add(store_in_map(true_deltaE_cut_module(track='TrueMuonTrack', dE=700, center=True, losses='TrueMuonLosses'), label='MCTCenterDeltaE700', map_name='DeltaECutMap'))
    tray.Add(counter(map_cut_module('DeltaECutMap')))

    # Compute energy histograms using first 600m of track
    reco_start_energy_collections = dict()
    tray.Add(energy_histogram_module(reco_start_energy_collections, pulse_series=pulse_series, track=reco_track, cut_maps=['GeoLengthCutMap', 'LengthCutMap', 'DeltaECutMap'], center=False))
    true_milli_start_energy_collections = dict()
    tray.Add(energy_histogram_module(true_milli_start_energy_collections, pulse_series=pulse_series, track='TrueMuonTrack', cut_maps=['GeoLengthCutMap', 'LengthCutMap', 'DeltaECutMap'], center=False))
    true_start_energy_collections = dict()
    tray.Add(true_energy_histogram_module(true_start_energy_collections, track='TrueMuonTrack', cut_maps=['GeoLengthCutMap', 'LengthCutMap', 'DeltaECutMap'], center=False, losses='TrueMuonTrack'))

    # Compute energy histograms using center 600m of track
    reco_center_energy_collections = dict()
    tray.Add(energy_histogram_module(reco_center_energy_collections, pulse_series=pulse_series, track=reco_track, cut_maps=['GeoLengthCutMap', 'LengthCutMap', 'DeltaECutMap'], center=True))
    true_milli_center_energy_collections = dict()
    tray.Add(energy_histogram_module(true_milli_center_energy_collections, pulse_series=pulse_series, track='TrueMuonTrack', cut_maps=['GeoLengthCutMap', 'LengthCutMap', 'DeltaECutMap'], center=True))
    true_center_energy_collections = dict()
    tray.Add(true_energy_histogram_module(true_start_energy_collections, track='TrueMuonTrack', cut_maps=['GeoLengthCutMap', 'LengthCutMap', 'DeltaECutMap'], center=True, losses='TrueMuonTrack'))

    # Compute track histograms after energy cuts
    reco_track_postcut_collections = dict()
    tray.Add(track_histogram_module(reco_track_postcut_collections, pulse_series=pulse_series, track=reco_track, cut_maps=['GeoLengthCutMap', 'LengthCutMap', 'DeltaECutMap']))
    true_track_postcut_collections = dict()
    tray.Add(track_histogram_module(true_track_postcut_collections, pulse_series=pulse_series, track='TrueMuonTrack', cut_maps=['GeoLengthCutMap', 'LengthCutMap', 'DeltaECutMap']))

    # Counter
    tray.Add(counter(lambda x: True, const=True))

    # Save the frames
    tray.AddModule('I3Writer', 'writer', filename=outfile, DropOrphanStreams=[icetray.I3Frame.DAQ])
    tray.AddModule('TrashCan','can')

    # Execute the tray
    print 'Executing tray'
    tray.Execute()
    tray.Finish()

    print counters

    # Assemble the histogram output
    output_dict = dict()
    output_dict['reco_track_collections'] = reco_track_collections
    output_dict['true_track_collections'] = true_track_collections
    output_dict['reco_start_energy_collections'] = reco_start_energy_collections
    output_dict['true_milli_start_energy_collections'] = true_milli_start_energy_collections
    output_dict['true_start_energy_collections'] = true_start_energy_collections
    output_dict['reco_center_energy_collections'] = reco_center_energy_collections
    output_dict['true_center_energy_collections'] = true_center_energy_collections
    output_dict['true_milli_center_energy_collections'] = true_milli_center_energy_collections
    output_dict['reco_track_postcut_collections'] = reco_track_postcut_collections
    output_dict['true_track_postcut_collections'] = true_track_postcut_collections

    # Store the computed histograms
    outpickle = open(hist_outfile, 'wb')
    pickle.dump(output_dict, outpickle, -1)
    outpickle.close()


if __name__ == '__main__':
    for pair in file_pairs:
        infiles, outfile = pair
        outfile_base = path.basename(outfile)
        outfile_base = outfile_base[:outfile_base.rfind('.i3')]
        hist_outfile = path.dirname(hist_file) + '/' + outfile_base + '_histograms.pkl'

        run_tray(geo, infiles, outfile, hist_outfile, pulse_series=_pulse_series, reco_track=_reco_track)

