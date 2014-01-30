"""Propagating 2D dynamics on alanine dipeptide using OpenMM.
"""
import os
import mdtraj as md
import mdtraj.reporters
from simtk import unit
import simtk.openmm as mm
from simtk.openmm import app
from SwitchingKalman import *
from MullerForce import *
from numpy import array, reshape, savetxt, loadtxt, zeros
from simtk.unit import kelvin, picosecond, femtosecond, nanometer, dalton
from Kmeans import *
import matplotlib.pyplot as pp
import numpy as np
from movies import *

SIMULATE = False
DELETE = True

LOAD = True
LEARN = True
MOVIE = True
NUM_TRAJS = 1
NUM_SCHED = 1
NUM_ITERS = 5
S = 10000
Delta = 10
T = S / Delta
em_vars = ['As', 'bs', 'Qs', 'Z']
K = 4

native = md.load('../data/native.pdb')
filenames = {}
filenames[10] = '../data/ala2_10.h5'
filenames[100] = '../data/ala2_100.h5'
filenames[1000] = '../data/ala2_1000.h5'
filenames[10000] = '../data/ala2_10000.h5'
out = '../data/metastable'
outs = []
for k in range(K):
  outs.append('metastable' + str(k))
if SIMULATE:
  topology = native.topology.to_openmm()
  forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
  system = forcefield.createSystem(topology,
              nonbondedMethod=app.CutoffNonPeriodic)
  integrator = mm.LangevinIntegrator(330*unit.kelvin,
      1.0/unit.picoseconds, 2.0*unit.femtoseconds)
  simulation = app.Simulation(topology, system, integrator)

  simulation.context.setPositions(native.xyz[0])
  simulation.context.setVelocitiesToTemperature(330*unit.kelvin)

  if DELETE and os.path.exists(filenames[T]):
    os.remove(filenames[T])
  if not os.path.exists(filenames[T]):
    simulation.reporters.append(md.reporters.HDF5Reporter(
      filenames[T], Delta))
    simulation.step(S)

if LOAD:
  filename_list = [v for (k,v) in filenames.items()]
  traj = md.load(filenames[T])
  traj.superpose(native)
  (T, N_atoms, dim) = shape(traj.xyz)
  y_dim = N_atoms
  x_dim = y_dim
  xs = distance_matrix(traj, native)
  gen_movie(xs, native, filename_list, '../data/alanine_true', N_atoms)

  if LEARN:
    means, assignments = kmeans(xs, K)
    bs = means
    Cs = zeros((K, y_dim, x_dim))
    Rs = zeros((K, y_dim, y_dim))
    for k in range(K):
      Cs[k] = reshape(eye(x_dim), (y_dim, x_dim))
      Rs[k] = reshape(0.01 * eye(y_dim), (y_dim, y_dim))
    l = SwitchingKalmanFilter(x_dim, y_dim, K=K, bs=bs, Cs=Cs, Rs=Rs)
    l.em(xs[:], em_iters=NUM_ITERS, em_vars=em_vars)
    sim_xs,sim_Ss = l.sample(T,s_init=0, x_init=means[0],
        y_init=means[0])
    gen_movie(sim_xs, native, filename_list,
        '../data/alanine_sim', N_atoms)
    metastable_states = l.compute_metastable_wells()
    gen_structures(metastable_states, native, filename_list, outs, N_atoms)
