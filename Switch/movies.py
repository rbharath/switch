"""Borrowed from mixtape's movie scripts"""
import os
import numpy as np
import mdtraj as md
from numpy import reshape, shape, zeros, arange
from numpy.linalg import norm

def load_timeseries(filenames, atom_indices, topology):
    X = []
    i = []
    f = []
    for file in filenames:
        t = md.load(file)
        t.superpose(topology, atom_indices=atom_indices)
        diff2 = (t.xyz[:, atom_indices] - topology.xyz[0, atom_indices])**2
        x = np.sqrt(np.sum(diff2, axis=2))

        X.append(x)
        i.append(np.arange(len(x)))
        f.extend([file]*len(x))
    return np.concatenate(X), np.concatenate(i), np.array(f)

def gen_movie(Ys, reference, filenames, out, T, N_atoms, dim):
  """Assumes that we have a full featurization of the model. Should fix
     this for larger systems.
  """
  atom_indices = arange(N_atoms)
  xx, ii, ff = load_timeseries(filenames, atom_indices, reference)
  movieframes = []
  for y in Ys:
    i = np.argmin(np.sum((y - xx)**2, axis=1))
    movieframes.append(md.load_frame(ff[i], ii[i]))
  movie = reduce(lambda a, b: a.join(b), movieframes)
  movie.superpose(movie)
  movie.save('%s.xtc' % out)
  movie[0].save('%s.xtc.pdb' % out)
  return movie, xx

def distance_matrix(traj, native):
  diff_traj = traj.superpose(native)
  diff = diff_traj.xyz
  (T, N_atoms, _) = shape(diff)
  dist= zeros((T, N_atoms))
  for t in range(T):
    for atom in range(N_atoms):
      dist[t, atom] = norm(diff[t,atom])
  return dist
