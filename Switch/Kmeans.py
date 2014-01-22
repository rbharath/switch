import numpy as np
from numpy import shape, zeros, Inf
from numpy.linalg import norm
from numpy.random import randint

def kmeans(ys, K):
  """ Takes a dataset and finds the K means through the usual
  k-means algorithm.
  Inputs:
    ys: Dataset of points
    K: number of means
  Outputs:
    means: Learned means
    assigments: Says which mean the t-th datapoint belongs to
  """
  (T, y_dim) = shape(ys)
  means = zeros((K, y_dim))
  old_means = zeros((K, y_dim))
  assignments = zeros(T)
  num_each = zeros(K)
  # Pick random observations as initializations
  for k in range(K):
    r = randint(0,T)
    means[k] = ys[r]
  Delta = Inf
  Epsilon = 1e-5
  iteration = 0
  while Delta >= Epsilon:
    #print "\tK-means iteration = %d" % iteration
    Delta = 0
    # Perform an Assignment Step
    for t in range(T):
      dist = Inf
      y = ys[t]
      # Find closest means
      for k in range(K):
        if norm(y - means[k]) < dist:
          dist = norm(y - means[k])
          assignments[t] = k
    #print "means[%d] = %s" % (iteration, str(means))
    # Perform Mean Update Step
    old_means[:] = means[:]
    # Reset means and num_each
    means[:] = 0
    num_each[:] = 0
    for t in range(T):
      k = assignments[t]
      num_each[k] += 1
      means[k] += ys[t]
    #print "num_each[%d] = %s" % (iteration, str(num_each))
    for k in range(K):
      means[k] /= num_each[k]
      Delta += norm(means[k] - old_means[k])
    # reset numeach
    iteration += 1
  return means, assignments

def means_match(base_means, means, assignments):
  (K, y_dim) = shape(means)
  (T,) = shape(assignments)
  matching = zeros(K)
  new_assignments = zeros(T)
  for i in range(K):
    closest = -1
    closest_dist = Inf
    for j in range(K):
      if norm(base_means[i] - means[j]) < closest_dist:
        closest = j
        print "base_means[%d] = %s" % (i, str(base_means[i]))
        print "means[%d] = %s" % (j, str(means[j]))
        closest_dist = norm(base_means[i], means[j])
    matching[i] = closest
  for t in range(T):
    new_assignments[t] = matching[assignments[t]]
  return matching, new_assignments
