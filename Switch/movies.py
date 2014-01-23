"""Borrowed from mixtape's movie scripts"""
import os
import numpy as np
import mdtraj as md
from numpy import reshape

def gen_movie(Ys, topology, out, T, N_atoms, dim):
  """Assumes that we have a full featurization of the model. Should fix
     this for larger systems.
  """
  Ys = reshape(Ys, (T, N_atoms, dim))
  traj = md.Trajectory(Ys, topology)
  #movieframes = []
  #for y in Ys:
  #  i = np.argmin(np.sum((y - xx)**2, axis=1))
  #  movieframes.append(md.load_frame(ff[i], ii[i]))
  #movie = reduce(lambda a, b: a.join(b), movieframes)
  #movie.superpose(movie)
  traj.save('%s.xtc' % out)
  traj[0].save('%s.xtc.pdb' % out)
