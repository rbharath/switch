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

def gen_movie(ys, reference, filenames, out, N_atoms):
  """Assumes that we have a full featurization of the model. Should fix
     this for larger systems.
  """
  atom_indices = arange(N_atoms)
  xx, ii, ff = load_timeseries(filenames, atom_indices, reference)
  movieframes = []
  for y in ys:
    i = np.argmin(np.sum((y - xx)**2, axis=1))
    movieframes.append(md.load_frame(ff[i], ii[i]))
  movie = reduce(lambda a, b: a.join(b), movieframes)
  movie.superpose(movie)
  movie.save('%s.xtc' % out)
  movie[0].save('%s.xtc.pdb' % out)

def gen_structures(ys, reference, filenames, outs, N_atoms):
  atom_indices = arange(N_atoms)
  xx, ii, ff = load_timeseries(filenames, atom_indices, reference)
  for y, out in zip(ys, outs):
    i = np.argmin(np.sum((y - xx)**2, axis=1))
    frame = md.load_frame(ff[i], ii[i])
    frame.superpose(reference)
    frame.save('%s.pdb' % out)

def distance_matrix(traj, native):
  traj.superpose(native)
  diff = (traj.xyz[:] - native.xyz[0])**2
  dist = np.sqrt(np.sum(diff, axis=2))
  return dist
