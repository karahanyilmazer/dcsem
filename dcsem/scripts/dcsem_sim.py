#!/usr/bin/env python

# dcsem_sim.py - wrapper script for running simulations
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2023 University of Oxford
# SHBASECOPYRIGHT

from dcsem.auxilary import configargparse as argparse

p = argparse.ArgumentParser("DCSEM - Simulator")

required = p.add_argument_group('required arguments')
optional = p.add_argument_group('additional options')


required.add_argument('--model', required=True, type=str, dest='model', metavar='<MODEL>',
                      help='Model to use. Either DCM or SEM')
required.add_argument('--num_rois', required=True, type=int,
                       help='Number of ROIs')
required.add_argument('--num_layers', required=True, type=int,
                      help='Number of Layers per ROI')
required.add_argument('--time_points', required=True, type=int,
                help='Number of Ltime points')
optional.add_argument('--tr', required=False, type=float, default=1.,
                help='Repetition time (seconds). Default = 1s')
optional.add_argument('--Amat', required=False, type=str,
                help='A matrix specification (text file)')
optional.add_argument('--Cmat', required=False, type=str,
                help='C matrix specification (text file)')
optional.add_argument('--sigma', required=False, type=float,default=1.,
                help='Value of Sigma for SEM')
optional.add_argument('--cnr', required=False, type=float,
                help='Value of CNR for DCM')
optional.add_argument('--self_conn', required=False, type=float,default=-1.,
                help='Value of Sigma for SEM')
optional.add_argument('--stim', required=False, type=str,
                help="Stimulus. Either 3-column format file, or 'random'")
optional.add_argument('--verbose', action='store_true', help='Verbose mode')
required.add_argument('--outdir', required=True, type=str,
                help='output folder')
optional.add('--config', required=False, is_config_file=True,
             help='configuration file')

args = p.parse_args()

# Begin

from dcsem import models, utils
import numpy as np
import os

# plotting function
def quickplot(ts, image_file):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ts)
    plt.savefig(image_file)


if args.model == 'SEM':
    if args.num_layers > 1:
        model = models.MultiLayerSEM(num_rois=args.num_rois, num_layers=args.num_layers)
    else:
        model = models.SEM(num_rois=args.num_rois)
elif args.model == 'DCM':
    if args.num_layers == 2:
        model = models.TwoLayerDCM(num_rois=args.num_rois)
    elif args.num_layers == 1:
        model = models.DCM(num_rois=args.num_rois)
    else:
        raise(Exception('DCM only implemented for 1 or 2 layers'))
else:
    raise(Exception('Unknown model'))

# A matrix
if args.Amat:
    A = utils.parse_matrix_file(args.Amat, args.num_rois, args.num_layers, args.self_conn)
    model.set_params({'A':A})

# C matrix (in case of DCM)
if args.Cmat and args.model == 'DCM':
    C = utils.parse_matrix_file(args.Amat, args.num_rois, args.num_layers)
    model.set_params({'C': C})

if args.verbose:
    print(model)

# Simulation
if args.verbose:
    print('begin simulation')
tvec = np.linspace(0, args.time_points * args.tr, args.time_points)
if args.model == 'SEM':
    sim = model.simulate(tvec)
else:
    # stimulus:
    if args.stim == 'random':
        u = utils.stim_random(tvec, args.num_rois*args.num_layers)
    else:
        u = utils.stim_boxcar(args.stim)
    sim = model.simulate(tvec, u = u, CNR=args.cnr)

# Save results here
if args.verbose:
    print('save results')

outdir = os.path.expanduser(args.outdir)
os.makedirs(outdir, exist_ok=True)

if args.model == 'SEM':
    np.savetxt(outdir+"/y.txt", sim)
    quickplot(sim, outdir+'/y.png')
else:
    np.savetxt(outdir+"/y.txt", sim['bold'])
    quickplot(sim['bold'], outdir+'/y.png')


# save more information
info = {}
info['model'] = model.model
info['num_rois'] = model.num_rois
info['num_layers'] = model.num_layers

import json
with open(outdir+'/info.json', "w") as outfile:
    json.dump(info, outfile)

