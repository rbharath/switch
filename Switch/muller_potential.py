"""Propagating 2D dynamics on the muller potential using OpenMM.

Currently, we just put a harmonic restraint on the z coordinate,
since OpenMM needs to work in 3D. This isn't really a big deal, except
that it affects the meaning of the temperature and kinetic energy. So
take the meaning of those numbers with a grain of salt.
"""
from SwitchingKalman import *
from MullerForce import *
from numpy import array, reshape, savetxt, loadtxt, zeros
from simtk.unit import kelvin, picosecond, femtosecond, nanometer, dalton
from Kmeans import *
import simtk.openmm as mm
import matplotlib.pyplot as pp
import numpy as np
import sys

PLOT = True
LEARN = True
NUM_TRAJS = 1

# each particle is totally independent
nParticles = 1
mass = 1.0 * dalton
#temps  = 200 300 500 750 1000 1250 1500 1750 2000
temperature = 2000 * kelvin
friction = 100 / picosecond
timestep = 10.0 * femtosecond
T = 300
sim_T = 1000

x_dim = 2
y_dim = 2
K = 3
NUM_ITERS = 3

em_vars = ['As', 'bs', 'Qs', 'Z']
As = zeros((K, x_dim, x_dim))
Sigmas = zeros((K, x_dim, x_dim))
Cs = zeros((K, y_dim, x_dim))
Rs = zeros((K, y_dim, y_dim))
for k in range(K):
  Cs[k] = reshape(eye(x_dim), (y_dim, x_dim))
  Rs[k] = reshape(0.01 * eye(y_dim), (y_dim, y_dim))
  Sigmas[k] = reshape(0.00001 * eye(x_dim), (x_dim, x_dim))

# Allocate Memory
start = T/4
ys = zeros((NUM_TRAJS * (T-start), y_dim))

# Clear Display
pp.cla()
# Choose starting conformations uniform on the grid
# between (-1.5, -0.2) and (1.2, 2)
########################################################################

for traj in range(NUM_TRAJS):
  system = mm.System()
  mullerforce = MullerForce()
  for i in range(nParticles):
    system.addParticle(mass)
    mullerforce.addParticle(i, [])
  system.addForce(mullerforce)

  integrator = mm.LangevinIntegrator(temperature, friction, timestep)
  context = mm.Context(system, integrator)
  startingPositions = (np.random.rand(nParticles, 3) * np.array([2.7, 1.8, 1])) + np.array([-1.5, -0.2, 0])

  context.setPositions(startingPositions)
  context.setVelocitiesToTemperature(temperature)

  trajectory = zeros((T,2))
  print "Traj %d" % traj
  for i in range(T):
    x = context.getState(getPositions=True).\
          getPositions(asNumpy=True).value_in_unit(nanometer)
    # Save the state
    #print "\tshape(x[%d]) = %s" % (i, str(shape(x)))
    if i > start:
      ys[traj * (T-start) + (i-start),:] = x[0,0:2]
    trajectory[i,:] = x[0,0:2]
    integrator.step(10)
  if PLOT:
    pp.plot(trajectory[start:,0], trajectory[start:,1], color='k')
    # Compute K-means
    means, assignments = kmeans(ys, K)
    pp.scatter(means[:,0], means[:,1], color='r',zorder=10)
    pp.scatter(ys[:,0], ys[:,1], edgecolor='none', facecolor='k',zorder=1)
    pp.show()

if LEARN:
  # Learn the Switching Filter
  bs = means
  l = SwitchingKalmanFilter(x_dim, y_dim, K=K, bs=bs, Cs=Cs, Rs=Rs)
  l.em(ys[:], em_iters=NUM_ITERS, em_vars=em_vars)
  sim_xs,sim_Ss,sim_ys = l.sample(sim_T,s_init=0, x_init=means[0], y_init=means[0])

if PLOT:
  Delta = 0.5
  minx = min(ys[:,0])
  maxx = max(ys[:,0])
  miny = min(ys[:,1])
  maxy = max(ys[:,1])
  if LEARN:
    minx = min(min(sim_ys[:,0]), minx) - Delta
    maxx = max(max(sim_ys[:,0]), maxx) + Delta
    miny = min(min(sim_xs[:,1]), miny) - Delta
    maxy = max(max(sim_xs[:,1]), maxy) + Delta
    pp.scatter(sim_ys[:,0], sim_ys[:,1], edgecolor='none',
        zorder=5,facecolor='g')
  MullerForce.plot(ax=pp.gca(),minx=minx,maxx=maxx,miny=miny,maxy=maxy)
