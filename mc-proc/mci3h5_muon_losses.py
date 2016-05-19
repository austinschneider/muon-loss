#!/usr/bin/env python
from __future__ import print_function

import sys
import itertools

from icecube import icetray, dataio
from icecube import dataclasses
from icecube import tableio, hdfwriter

from I3Tray import *

tray = I3Tray()

#'/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.i3.bz2'
#tray.Add("I3Reader", "my_reader", FilenameList=['GeoCalibDetectorStatus_2012.56063_V1.i3.gz','/scratch/aschneider/Level2_IC86.2012_nugen_numu.011374.000050.clsim-base-4.0.3.0.99_eff.i3.bz2'])


infiles = ['/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11374/00000-00999/clsim-base-4.0.3.0.99_eff/Level2_IC86.2012_nugen_numu.011374.%06.0f.clsim-base-4.0.3.0.99_eff.i3.bz2' % i for i in range(50, 61)]

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
        #print(len(primaries))
        if(len(primaries)>1):
            print("More than one neutrino primary! Using max energy neutrino for frame.")
            primaries = [max(primaries, key=self.get_energy)]
        daughters = [Tree.get_daughters(p.id) for p in primaries] #Get the daughters of the primaries
        muons = [[d for d in p if(d.type == d.MuPlus or d.type == d.MuMinus)] for p in daughters] #Select only daughters that are muons
        highEMuons = [max(p, key=self.get_energy) if len(p) > 0 else None for p in muons] #Choose only the highest energy muon
        highEMuons = [(m if abs(m.pos) < 500 else None) if m != None else None for m in highEMuons] #Choose only muons that start within 500m of the detector center
        losses = [[d for d in Tree.get_daughters(m) if d.type in self.loss_set] if m != None else None for m in highEMuons] #Get the muon daughters
        #print(len([m for m in highEMuons if m != None]))

        if(len(muons) > 1):
            raise ValueError('There is more than one muon in the frame.')

        HighEMuonLosses = dataclasses.I3VectorI3Particle()
        #for p,hEM,loss in itertools.izip(primaries, highEMuons, losses):
        #    if hEM != None:
        #        HighEMuonLosses.append(hEM)
        if highEMuons[0] != None:
            HighEMuonLosses.append(highEMuons[0])
            for loss in losses[0]:
                HighEMuonLosses.append(loss)
        frame['HighEMuonLosses'] = HighEMuonLosses

        self.PushFrame(frame)

outfile = "out.h5"
hdftable = hdfwriter.I3HDFTableService(outfile)

tray.Add("I3Reader", "my_reader", FilenameList=infiles)

tray.Add(lambda frame: frame.Has('I3MCTree'))

tray.Add(MyModule)

tray.Add(tableio.I3TableWriter,'hdf1',
         tableservice = hdftable,
         SubEventStreams = ['InIce', 'InIceSplit'],
         keys = ['HighEMuonLosses']
        )

tray.Execute()
tray.Finish()
