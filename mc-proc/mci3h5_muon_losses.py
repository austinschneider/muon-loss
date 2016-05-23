#!/usr/bin/env python
from __future__ import print_function

import sys
import itertools
import glob
import ntpath

from icecube import icetray, dataio
from icecube import dataclasses
from icecube import tableio, hdfwriter

from I3Tray import *

tray = I3Tray()

#'/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.i3.bz2'
#tray.Add("I3Reader", "my_reader", FilenameList=['GeoCalibDetectorStatus_2012.56063_V1.i3.gz','/scratch/aschneider/Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.i3.bz2'])


#infiles = ['/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_numu.011374.%06.0f.clsim-base-4.0.3.0.99_eff.i3.bz2' % i for i in range(50, 61)]

import argparse

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
    files = glob.glob('%s*.h5' % dir)
    for f in files:
        file_source = ntpath.splitext(ntpath.basename(f))[0] + '.i3.bz2'
        if('-' in file_source.split('.')[3]):
            source_nums = (lambda x: range(int(x[0]), int(x[1])+1))(file_source.split('.')[3].split('-')) #Filename has number after 3rd period
            for n in source_nums:
                source_split = file_source.split('.')
                source_split[3] = '%06.0f' % n #Filename has number after third period
                f_source = '.'.join(source_split)
                processed_files.add(f_source)
        else:
            processed_files.add(file_source)
                                                                               
    return processed_files

# Store 3 kinds of particles
#   Primary neutrino
#   Highest energy muon daughter
#   Losses of that muon
class MyModule(icetray.I3ConditionalModule):
    def __init__(self, context):
        super(MyModule, self).__init__(context)
    def get_energy(self, particle):
        return particle.energy
    def Configure(self):
        p = dataclasses.I3Particle()
        self.loss_set = set([p.PairProd, p.DeltaE, p.Brems, p.NuclInt])
        self.nu_set = set([p.Nu, p.NuE, p.NuEBar, p.NuMu, p.NuMuBar, p.NuTau, p.NuTauBar])
    def DAQ(self, frame):
        Tree = frame['I3MCTree']
        primaries = Tree.primaries
        primaries = [p for p in primaries if p.type in self.nu_set]
        if(len(primaries)>1):
            #Not the correct way to handle this. Don't expect this to happen so should throw an error
            print("More than one neutrino primary! Using max energy neutrino for frame.")
            primaries = [max(primaries, key=self.get_energy)]
        daughters = [Tree.get_daughters(p.id) for p in primaries] #Get the daughters of the primaries
        muons = [[d for d in p if(d.type == d.MuPlus or d.type == d.MuMinus)] for p in daughters] #Select only daughters that are muons
        highEMuons = [max(p, key=self.get_energy) if len(p) > 0 else None for p in muons] #Choose only the highest energy muon
        highEMuons = [(m if abs(m.pos) < 500 else None) if m != None else None for m in highEMuons] #Choose only muons that start within 500m of the detector center
        highEMuons = [(m if m.energy >= 5000 else None) if m != None else None for m in highEMuons] #Choose only muons that are at least 5TeV
        losses = [[d for d in Tree.get_daughters(m) if d.type in self.loss_set] if m != None else None for m in highEMuons] #Get the muon daughters

        if(len(muons) > 1):
            raise ValueError('There is more than one muon in the frame.')

        HighEMuonLosses = dataclasses.I3VectorI3Particle()
        if highEMuons[0] != None:
            HighEMuonLosses.append(primaries[0])
            HighEMuonLosses.append(highEMuons[0])
            for loss in losses[0]:
                HighEMuonLosses.append(loss)
        else:
            del frame['I3MCWeightDict']
        frame['HighEMuonLosses'] = HighEMuonLosses

        self.PushFrame(frame)

# Only look at files not already handled
processed_files = get_processed_files(outdir)
infiles = [file for indir in indirs for file in glob.glob('%s*.i3.bz2' % indir) if ntpath.basename(file) not in processed_files]

infiles.sort()

while(len(infiles)):
    try:
        # Pick the first N files to process
        bundle = infiles[:bundle_size]
        name_split = ntpath.basename(bundle[0]).split('.')
        name_split[3]  = '%06.0f-%06.0f' % (lambda x: (min(x), max(x)))([int(ntpath.basename(infile).split('.')[3]) for infile in bundle])
        outfile = outdir + '.'.join(name_split[:-2]) + '.h5'

        tray = I3Tray()
        hdftable = hdfwriter.I3HDFTableService(outfile)
        tray.Add("I3Reader", "my_reader", FilenameList=bundle)
        tray.Add(lambda frame: frame.Has('I3MCTree'))
        tray.Add(MyModule)
        
        tray.Add(tableio.I3TableWriter,'hdf1',
                 tableservice = hdftable,
                 SubEventStreams = ['InIce', 'InIceSplit'],
                 keys = ['HighEMuonLosses', 'I3MCWeightDict']
                )
        
        tray.Execute()
        tray.Finish()
    # Skip I3 files that have issues
    except RuntimeError as e:
        pass
    
    processed_files = get_processed_files(outdir)
    infiles = [file for indir in indirs for file in glob.glob('%s*.i3.bz2' % indir) if ntpath.basename(file) not in processed_files]
    infiles.sort()
    #bundle_start += bundle_size
