#!/usr/bin/env python

# dcsem_sim.py - wrapper script for running simulations
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2023 University of Oxford
# SHBASECOPYRIGHT

import argparse

p = argparse.ArgumentParser("DCSEM - Simulator")
p.add_argument('--model', required=True, type=str, dest='model',
                   metavar='<MODEL>', help='Model to use. Either DCM or SEM')
p.add_argument('--num_rois', required=True, type=int,
                help='Number of ROIs')
p.add_argument('--num_layers', required=True, type=int,
                help='Number of Layers per ROI')
p.add_argument('--time_points', required=True, type=int,
                help='Number of Ltime points')
p.add_argument('--tr', required=False, type=float, default=1.,
                help='Repetition time (seconds). Default = 1s')
p.add_argument('--Amat', required=False, type=str,
                help='A matrix specification (text file)')
p.add_argument('--Cmat', required=False, type=int,
                help='C matrix specification (text file)')
p.add_argument('--sigma', required=False, type=float,default=1.,
                help='Value of Sigma for SEM')
p.add_argument('--cnr', required=False, type=float,
                help='Value of CNR for DCM')
p.add_argument('--self_conn', required=False, type=float,default=-1,
                help='Value of Sigma for SEM')
p.add_argument('--stimulus', required=False, type=str,
                help="Stimulus. Either 3-column format file, or 'random'")
p.add_argument('--verbose', action='store_true', help='Verbose mode')
p.add_argument('--outdir', required=True, type=str,
                help='output folder')

args = p.parse_args()

# Begin

from dcsem import models, utils
import numpy as np

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
    conn = []
    with open(args.Amat, 'r') as f:
        conn = [l.rstrip() for l in f]

    A = utils.create_A_matrix(args.num_rois, args.num_layers, conn)

    if args.model == 'DCM':
        A = utils.create_A_matrix(args.num_rois, args.num_layers, conn, args.self_conn)
    model.set_params({'A':A})

# C matrix (in case of DCM)
if args.Cmat:
    conn = []
    with open(args.Cmat, 'r') as f:
        conn = [l.rstrip() for l in f]
    if args.model == 'DCM':
        C = utils.create_A_matrix(args.num_rois, args.num_layers, conn)
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
        u = utils.stim_random(tvec, args.num_rois)
    else:
        u = utils.stim_boxcar(args.stim)
    sim = model.simulate(tvec, u = u, CNR=args.cnr)

# Save results here
if args.verbose:
    print('save results')

import os
os.makedirs(args.outdir, exist_ok=True)

if args.model == 'SEM':
    np.savetxt(args.outdir+"/y.txt", sim)

# save more information
import pandas as pd
info = {}
info['model'] = model.model
info['num_rois'] = model.num_rois
info['num_layers'] = model.num_layers

import json

with open(args.outdir+'/info.json', "w") as outfile:
    json.dump(info, outfile)

