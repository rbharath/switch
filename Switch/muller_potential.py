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
temperature = 3000 * kelvin
friction = 100 / picosecond
timestep = 10.0 * femtosecond
T = 500
sim_T = 1000

x_dim = 2
y_dim = 2
K = 3
NUM_ITERS = 5

em_vars = ['As', 'bs', 'Qs', 'Z', 'mus', 'Sigmas']
As = zeros((K, x_dim, x_dim))
bs = zeros((K, x_dim))
mus = zeros((K, x_dim))
Sigmas = zeros((K, x_dim, x_dim))
Qs = zeros((K, x_dim, x_dim))

# Allocate Memory
start = T/4
xs = zeros((NUM_TRAJS * (T-start), y_dim))

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
      xs[traj * (T-start) + (i-start),:] = x[0,0:2]
    trajectory[i,:] = x[0,0:2]
    integrator.step(10)
  if PLOT:
    pp.plot(trajectory[start:,0], trajectory[start:,1], color='k')
    # Compute K-means
    means, assignments = kmeans(xs, K)
    pp.scatter(means[:,0], means[:,1], color='r',zorder=10)
    pp.scatter(xs[:,0], xs[:,1], edgecolor='none', facecolor='k',zorder=1)
    pp.show()

if LEARN:
  # Compute K-means
  means, assignments = kmeans(xs, K)
  W_i_Ts = assignment_to_weights(assignments,K)
  emp_means, emp_covars = empirical_wells(xs, W_i_Ts)
  for i in range(K):
    A = randn(x_dim, x_dim)
    u, s, v = svd(A)
    As[i] = rand() * dot(u, v.T)
    bs[i] = dot(eye(x_dim) - As[i], means[i])
    mus[i] = emp_means[i]
    Sigmas[i] = emp_covars[i]
    Qs[i] = 0.5 * Sigmas[i]

  # Learn the Switching Filter
  bs = means
  l = SwitchingKalmanFilter(x_dim, y_dim, K=K,
      As=As,bs=bs,mus=mus,Sigmas=Sigmas,Qs=Qs)
  l.em(xs[:], em_iters=NUM_ITERS, em_vars=em_vars)
  sim_xs,sim_Ss = l.sample(sim_T,s_init=0, x_init=means[0], y_init=means[0])

if PLOT:
  Delta = 0.5
  minx = min(xs[:,0])
  maxx = max(xs[:,0])
  miny = min(xs[:,1])
  maxy = max(xs[:,1])
  if LEARN:
    minx = min(min(sim_xs[:,0]), minx) - Delta
    maxx = max(max(sim_xs[:,0]), maxx) + Delta
    miny = min(min(sim_xs[:,1]), miny) - Delta
    maxy = max(max(sim_xs[:,1]), maxy) + Delta
    pp.scatter(sim_xs[:,0], sim_xs[:,1], edgecolor='none',
        zorder=5,facecolor='g')
    pp.plot(sim_xs[:,0], sim_xs[:,1], zorder=5,color='g')
  MullerForce.plot(ax=pp.gca(),minx=minx,maxx=maxx,miny=miny,maxy=maxy)
