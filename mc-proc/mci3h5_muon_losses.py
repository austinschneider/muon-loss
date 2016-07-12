#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import itertools
import glob
import ntpath
import pickle

import numpy as np

from icecube import icetray, dataio
from icecube import dataclasses
from icecube import tableio, hdfwriter
from icecube import simclasses
from icecube import NewNuFlux
from icecube.icetray import I3Units
from icecube.weighting.weighting import from_simprod

from I3Tray import *
import argparse

# Stores simulation information for muon neutrinos, highest energy daughter muon, and losses of the muon
#   Primary: store I3Particle of the primary neutrino
#   Track: store I3MMCTrackList (only contains one element, but must be stored as a vector because I3MMCTrack is not a frame object)
#   Losses: store I3VectorI3Particle that contains all "stochastic" losses for the muon (DeltaE are ionization losses that are large enough that the simulation considers them stochastic, but they should be treated as continuous for the purposes of plotting)
# Only keeps muons of at least 5TeV at creation
# Only keeps muons that begin within 500m of the detector
# Saves all information to an h5 file


# Example directory:
# /data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/
# Example file:
# /data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.i3.bz2

# Initialize arg parser
parser = argparse.ArgumentParser(description='Proccess filenames')

parser.add_argument('indirs', metavar='indirs',  nargs='+')
parser.add_argument('-g', '--geo', default='')
parser.add_argument('-o', '--outdir', default='')
parser.add_argument('-b', '--bundlesize', type=int, default=10)

args = parser.parse_args()
indirs = args.indirs
geo = args.geo
outdir = args.outdir
bundle_size = args.bundlesize

# Get files aleady processed by looking at the output directory
# Output files have same similar filename structure to input files
# Difference is output files have number range
def get_processed_files(dir):
    processed_files = set()
    #files = glob.glob('%s*.h5' % dir)
    files = glob.glob('%s*.pkl' % dir)
    for f in files:
        file_source = ntpath.splitext(ntpath.basename(f))[0] + '.i3.bz2'
        if('-' in file_source.split('.')[3]): # If there is a number range
            source_nums = (lambda x: range(int(x[0]), int(x[1])+1))(file_source.split('.')[3].split('-')) # Filename has number after 3rd period
            for n in source_nums:
                source_split = file_source.split('.')
                source_split[3] = '%06.0f' % n #Filename has number after third period
                f_source = '.'.join(source_split)
                processed_files.add(f_source)
        else:
            processed_files.add(file_source)
                                                                               
    return processed_files

# Get input files that have no already been processed
def get_infiles(indirs, outdir):
    processed_files = get_processed_files(outdir)
    infiles = [file for indir in indirs for file in glob.glob('%s*.i3.bz2' % indir) if ntpath.basename(file) not in processed_files]
    infiles.sort()
    return infiles
    

# Define particle sets
p = dataclasses.I3Particle()
loss_set = set([p.PairProd, p.DeltaE, p.Brems, p.NuclInt])
nu_set = set([p.Nu, p.NuE, p.NuEBar, p.NuMu, p.NuMuBar, p.NuTau, p.NuTauBar])
mu_set = set([p.MuPlus, p.MuMinus])

# Define global variables
muon_checkpoints = []
muon_losses = []
weights = []

run_id = []
event_id = []

mu_info = []
nu_info = []

flux = NewNuFlux.makeFlux('honda2006').getFlux

sim_vol_length = 1600
sim_vol_radius = 800

sim_vol_top = sim_vol_length / 2
sim_vol_bottom = -sim_vol_top

is_in_sim_vol = lambda pos: (pos.z < sim_vol_top and pos.z > sim_vol_bottom and pos.rho < sim_vol_radius)

#generator = from_simprod(11374)
generator = from_simprod(10602)

# Store 3 kinds of particles
#   Primary neutrino
#   Highest energy muon daughter
#   Losses of that muon
class MyModule(icetray.I3ConditionalModule):
    def __init__(self, context):
        super(MyModule, self).__init__(context)
        self.keep_DAQ = False
    def get_energy(self, track):
        return track.particle.energy
    def Configure(self):
        pass
    def Physics(self, frame):
        pass
    def DAQ(self, frame):
        self.keep_DAQ = False

        Tree = frame['I3MCTree']
        tracks = frame['MMCTrackList']
        weight_dict = frame['I3MCWeightDict']
        header = frame['I3EventHeader']
        Tree_parent = Tree.parent

        primaries = Tree.primaries

        tracks = [track for track in tracks if track.particle.type in mu_set] # Only tracks that are muons
        tracks = [track for track in tracks if Tree.has(track.particle)] # Only tracks in MC tree
        tracks = [track for track in tracks if Tree_parent(track.particle).type in nu_set] # Only tracks that have nu parent
        tracks = [track for track in tracks if Tree_parent(track.particle) in primaries] # Only tracks that have primary parent
        tracks = [track for track in tracks if track.particle.energy >= 5000] # Only muons that are at least 5TeV at creation
        
        nu_primaries_of_tracks = np.unique([Tree_parent(track.particle) for track in tracks])
        tracks_by_primary = [[track for track in tracks if Tree_parent(track.particle) == p] for p in nu_primaries_of_tracks]
        max_E_muons_by_primary = [max(tracks_for_primary, key=self.get_energy) for tracks_for_primary in tracks_by_primary]

        n_muons = len(max_E_muons_by_primary)
        
        if(n_muons > 1):
            raise ValueError('There is more than one muon in the frame')
        
        #HighEMuonLosses = dataclasses.I3VectorI3Particle()
        #HighEMuonTracks = simclasses.I3MMCTrackList()
        if(n_muons > 0):
            muon_track = max_E_muons_by_primary[0]
            nu = nu_primaries_of_tracks[0]
            muon_p = muon_track.particle
            #HighEMuonTracks.append(muon_track)
            losses = [d for d in Tree.get_daughters(muon_p) if d.type in loss_set] #Get the muon daughters

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
            nu_cos_zenith = np.cos(nu.dir.zenith)
            nu_p_int = weight_dict['TotalInteractionProbabilityWeight']
            nu_unit = I3Units.cm2/I3Units.m2
            nu_weight = nu_p_int*(flux(nu.type, nu.energy, nu_cos_zenith)/nu_unit)/generator(nu.energy, nu.type, nu_cos_zenith)

            muon_losses.append(loss_tuples)
            weights.append(nu_weight)
            muon_checkpoints.append(checkpoints)

            mu_info.append((muon_p.energy, muon_p.pos.x, muon_p.pos.y, muon_p.pos.z, muon_p.dir.zenith, muon_p.dir.azimuth, muon_p.length))
            nu_info.append((nu.energy, nu.pos.x, nu.pos.y, nu.pos.z, nu.dir.zenith, nu.dir.azimuth, nu.length))

            run_id.append(header.run_id)
            event_id.append(header.event_id)
            
            #for loss in losses:
            #    HighEMuonLosses.append(loss)
            #frame['Primary'] = Tree_parent(muon_track.particle)
            #frame['Track'] = HighEMuonTracks
            #frame['Losses'] = HighEMuonLosses
            
        #self.PushFrame(frame)

infiles = get_infiles(indirs, outdir)

bad_files = set()

e = []

while(len(infiles)):
    try:
        # Pick the first N files to process
        bundle = infiles[:bundle_size]
        name_split = ntpath.basename(bundle[0]).split('.')
        name_split[3]  = '%06.0f-%06.0f' % (lambda x: (min(x), max(x)))([int(ntpath.basename(infile).split('.')[3]) for infile in bundle]) # File number after third period
        #outfile = outdir + '.'.join(name_split[:-2]) + '.h5' # Remove the two extensions and replace with .h5
        outfile = outdir + '.'.join(name_split[:-2]) + '.pkl' # Remove the two extensions and replace with .pkl

        os.system('touch %s' % outfile)
        
        # Clear the data
        muon_losses = []
        weights = []
        muon_checkpoints = []
        mu_info = []
        nu_info = []
        
        tray = I3Tray()
        #hdftable = hdfwriter.I3HDFTableService(outfile)
        tray.Add("I3Reader", "my_reader", FilenameList=bundle)
        tray.Add(lambda frame: frame.Has('I3MCTree'))
        tray.Add(lambda frame: frame.Has('I3MCWeightDict'))
        tray.Add(lambda frame: frame.Has('MMCTrackList'))
        tray.Add(MyModule)
        
        #tray.Add(tableio.I3TableWriter,'hdf1',
        #         tableservice = hdftable,
        #         SubEventStreams = ['InIce', 'InIceSplit'],
        #         keys = ['Primary', 'Track', 'Losses', 'I3MCWeightDict']
        #        )
        
        tray.Execute()
        # Pickle the data
        outpickle = open(outfile, 'wb')
        pickle.dump(muon_losses, outpickle, -1)
        pickle.dump(weights, outpickle, -1)
        pickle.dump(muon_checkpoints, outpickle, -1)
        pickle.dump(mu_info, outpickle, -1)
        pickle.dump(nu_info, outpickle, -1)
        pickle.dump(run_id, outpickle, -1)
        pickle.dump(event_id, outpickle, -1)
        outpickle.close()
        
        tray.Finish()

    # Skip I3 files that have issues
    except RuntimeError as e:
        #try:
        #    bad_file = e.message.split('Error reading')[1].split('at frame')[0].strip(' ')
        #except:
        #    bad_file = ''
        #    pass
        #bad_files.add(bad_file)
        for bad_file in bundle:
            bad_files.add(bad_file)
        #pass
    
    infiles = [file for file in get_infiles(indirs, outdir) if file not in bad_files]

