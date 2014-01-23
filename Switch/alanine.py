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
LOAD = True
LEARN = True
MOVIE = True
NUM_TRAJS = 1
NUM_SCHED = 1
NUM_ITERS = 2

if SIMULATE:
  pdb = md.load('native.pdb')
  topology = pdb.topology.to_openmm()
  forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
  system = forcefield.createSystem(topology,
              nonbondedMethod=app.CutoffNonPeriodic)
  integrator = mm.LangevinIntegrator(330*unit.kelvin,
      1.0/unit.picoseconds, 2.0*unit.femtoseconds)
  simulation = app.Simulation(topology, system, integrator)

  simulation.context.setPositions(pdb.xyz[0])
  simulation.context.setVelocitiesToTemperature(330*unit.kelvin)

  if not os.path.exists('ala2.h5'):
    simulation.reporters.append(md.reporters.HDF5Reporter('ala2.h5', 10))
    simulation.step(1000)
if LOAD:
  traj_filename = 'ala2.h5'
  traj = md.load(traj_filename)
  topology = traj.topology
  (T, N_atoms, dim) = shape(traj.xyz)
  y_dim = N_atoms * dim
  x_dim = y_dim
  ys = reshape(traj.xyz, (T, y_dim))

if LEARN:
  em_vars = ['As', 'bs', 'Qs', 'Z', 'pi']
  K = 2
  means, assignments = kmeans(ys, K)
  bs = means
  Cs = zeros((K, y_dim, x_dim))
  Rs = zeros((K, y_dim, y_dim))
  for k in range(K):
    Cs[k] = reshape(eye(x_dim), (y_dim, x_dim))
    Rs[k] = reshape(0.01 * eye(y_dim), (y_dim, y_dim))
  sim_T = 50
  l = SwitchingKalmanFilter(x_dim, y_dim, K=K, bs=bs, Cs=Cs, Rs=Rs)
  (out_x_tTs, out_x_tts, out_x_j_tTs, out_x_j_tts,
    out_V_j_tts, out_log_ll_j_t, out_log_ll_t, out_W_i_Ts,
    mus_iter, Sigmas_iter, As_iter, bs_iter, Qs_iter, Cs_iter,
    Rs_iter, pi_iter, Z_iter) = \
        l.daem(ys[:], em_num_sched=NUM_SCHED,
            em_iters=NUM_ITERS, em_vars=em_vars)
  sim_xs,sim_Ss,sim_ys = l.simulate(sim_T,s_init=0, x_init=means[0],
      y_init=means[0])
  gen_movie(sim_ys, topology, 'alanine', sim_T, N_atoms, dim)
  gen_movie(out_x_tTs[NUM_SCHED-1,NUM_ITERS-1],
      topology, 'alanine2', T, N_atoms, dim)
