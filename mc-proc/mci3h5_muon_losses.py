#!/usr/bin/env python
from __future__ import print_function

import pdb

import sys
import itertools
import glob
import ntpath

from icecube import icetray, dataio
from icecube import dataclasses
from icecube import tableio, hdfwriter
from icecube import simclasses

from I3Tray import *

tray = I3Tray()

#'/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.i3.bz2'
#tray.Add("I3Reader", "my_reader", FilenameList=['GeoCalibDetectorStatus_2012.56063_V1.i3.gz','/scratch/aschneider/Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.i3.bz2'])


#infiles = ['/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_numu.011374.%06.0f.clsim-base-4.0.3.0.99_eff.i3.bz2' % i for i in range(50, 61)]

import argparse

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

def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Get files aleady processed by looking at the output directory
# Output files have same similar filename structure to input files
# Difference is output files have number range
def get_processed_files(dir):
    processed_files = set()
    files = glob.glob('%s*.h5' % dir)
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

# Store 3 kinds of particles
#   Primary neutrino
#   Highest energy muon daughter
#   Losses of that muon
class MyModule(icetray.I3ConditionalModule):
    def __init__(self, context):
        super(MyModule, self).__init__(context)
    def get_energy(self, track):
        return track.particle.energy
    def Configure(self):
        pass
    def Physics(self, frame):
        Tree = frame['I3MCTree']
        tracks = frame['MMCTrackList']
        Tree_parent = Tree.parent

        primaries = Tree.primaries

        tracks = [track for track in tracks if track.particle.type in mu_set] # Only tracks that are muons
        tracks = [track for track in tracks if Tree.has(track.particle)] # Only tracks in MC tree
        tracks = [track for track in tracks if Tree_parent(track.particle).type in nu_set] # Only tracks that have nu parent
        tracks = [track for track in tracks if Tree_parent(track.particle) in primaries] # Only tracks that have primary parent
        
        tracks = [track for track in tracks if track.particle.energy > 5000] # Only muons that are at least 5TeV at creation
        tracks = [track for track in tracks if abs(track.particle.pos) < 500] # Only muons that are created within 500m of the detector

        nu_primaries_of_tracks = unique_list([Tree_parent(track.particle) for track in tracks])
        tracks_by_primary = [[track for track in tracks if Tree_parent(track.particle) == p] for p in nu_primaries_of_tracks]
        max_E_muons_by_primary = [max(tracks_for_primary, key=self.get_energy) for tracks_for_primary in tracks_by_primary]

        n_muons = len(max_E_muons_by_primary)
        
        if(n_muons > 1):
            raise ValueError('There is more than one muon in the frame')
        
        HighEMuonLosses = dataclasses.I3VectorI3Particle()
        HighEMuonTracks = simclasses.I3MMCTrackList()
        if(n_muons > 0):
            muon_track = max_E_muons_by_primary[0]
            losses = [d for d in Tree.get_daughters(muon_track.particle) if d.type in loss_set] #Get the muon daughters
            for loss in losses:
                HighEMuonLosses.append(loss)
            frame['Primary'] = Tree_parent(muon_track.particle)
            frame['Track'] = HighEMuonTracks
            frame['Losses'] = HighEMuonLosses
            self.PushFrame(frame)

# Only look at files not already handled
infiles = get_infiles(indirs, outdir)

while(len(infiles)):
    try:
        # Pick the first N files to process
        bundle = infiles[:bundle_size]
        name_split = ntpath.basename(bundle[0]).split('.')
        name_split[3]  = '%06.0f-%06.0f' % (lambda x: (min(x), max(x)))([int(ntpath.basename(infile).split('.')[3]) for infile in bundle]) # File number after third period
        outfile = outdir + '.'.join(name_split[:-2]) + '.h5' # Remove the two extensions and replace with .h5

        tray = I3Tray()
        hdftable = hdfwriter.I3HDFTableService(outfile)
        tray.Add("I3Reader", "my_reader", FilenameList=bundle)
        tray.Add(lambda frame: frame.Has('I3MCTree'))
        tray.Add(lambda frame: frame.Has('I3MCWeightDict'))
        tray.Add(lambda frame: frame.Has('MMCTrackList'))
        tray.Add(MyModule)
        
        tray.Add(tableio.I3TableWriter,'hdf1',
                 tableservice = hdftable,
                 SubEventStreams = ['InIce', 'InIceSplit'],
                 keys = ['Primary', 'Track', 'Losses', 'I3MCWeightDict']
                )
        
        tray.Execute()
        tray.Finish()

    # Skip I3 files that have issues
    except RuntimeError as e:
        pass
    
    infiles = get_infiles(indirs, outdir)

