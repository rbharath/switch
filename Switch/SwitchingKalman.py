import numpy as np
from numpy import dot, shape, eye, outer, sqrt, sum, log, diagonal, zeros
from numpy import nonzero, exp, shape, reshape, diag, maximum, isnan
from numpy import copy, minimum, ones, newaxis, mod, Inf
from numpy.linalg import svd, inv, eig
from numpy.random import randint, randn, multinomial, multivariate_normal
from numpy.random import rand
from Kmeans import *
from utils import *
import scipy.linalg as linalg
import scipy.stats as stats
import sys
"""
An Implementation of the Switching Kalman Filter. Uses the Generalized-
Pseudo-Bayesian algorithm of order 2, which collapses the exponentially
large posterior to finitely many Gaussians through moment matching, to
perform smooothing on the hidden real-valued states. A forward-backward
inference pass computes switch posteriors from the smoothed hidden states.
The switch posteriors are used in the M-step to update parameter estimates.
@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""

class SwitchingKalmanFilter(object):
  """Implements a Switching Kalman Filter along with an EM algorithm.
  """
  def __init__(self, x_dim, y_dim, K=None, As=None, bs=None, Qs=None,
      Z=None, mus=None, Sigmas=None):
    """
    Inputs:
      x_dim: dimension of hidden state
      y_dim: dimension of observable state
      K: Number of switching states.
      As: System transition matrices for each switching state.
      Qs: System covariance matrices for each switching state.
      Z: The switching matrix for the discrete state
      mus: the means for each metastable state
      Sigmas: the covariance for each metastable state
    """
    self.x_dim = x_dim
    self.y_dim = y_dim
    if K == None:
      K = 1
    self.K = K
    if As != None and shape(As) == (K, x_dim, x_dim):
      self.As = copy(As)
    else:
      self.As = zeros((K,x_dim,x_dim))
      for i in range(K):
        A = randn(x_dim, x_dim)
        # Stabilize A
        u,s,v = svd(A)
        self.As[i] = dot(u, v.T)
    if bs != None and shape(bs) == (K, x_dim):
      self.bs = copy(bs)
    else:
      self.bs = randn(K,x_dim)
    if Qs != None and shape(Qs) == (K, x_dim, x_dim):
      self.Qs = copy(Qs)
    else:
      self.Qs = zeros((K,x_dim, x_dim))
      for i in range(K):
        r = rand(x_dim, x_dim)
        r = (1.0/x_dim) * dot(r.T,r)
        self.Qs[i] = r
    if Z != None and shape(Z) == (K,K):
      self.Z = copy(Z)
    else:
      self.Z = rand(K,K)
      self.Z = self.Z / (sum(self.Z,axis=0))
    if mus != None and shape(mus) == (K,x_dim):
      self.mus = copy(mus)
    else:
      self.mus = zeros((K,x_dim))
      for i in range(K):
        self.mus[i] = randn(x_dim)
    if Sigmas != None and shape(Sigmas) == (K, x_dim, x_dim):
      self.Sigmas = copy(Sigmas)
    else:
      self.Sigmas = zeros((K,x_dim, x_dim))
      for i in range(K):
        r = rand(x_dim, x_dim)
        r = dot(r,r.T)
        self.Sigmas[i] = 0.1 * eye(x_dim) + r
    if bs != None and shape(bs) == (K, x_dim):
      self.bs = copy(bs)
    else:
      self.bs = zeros((K, x_dim))
      for i in range(K):
        b = rand(x_dim)
        self.bs[i] = b

  def sample(self, T, s_init=None,x_init=None,y_init=None):
    """
    Inputs:
      T: time to run simulation
    Outputs:
      xs: Hidden continuous states
      Ss: Hidden switch states
    """
    x_dim, y_dim, = self.x_dim, self.y_dim
    # Allocate Memory
    xs = zeros((T, x_dim))
    Ss = zeros(T)
    # Compute Invariant
    _, vl = linalg.eig(self.Z, left=True, right=False)
    pi = vl[:,0]
    # Sample Start conditions
    sample = multinomial(1, pi, size=1)
    if s_init == None:
      Ss[0] = nonzero(sample)[0][0]
    else:
      Ss[0] = s_init
    if x_init == None:
      xs[0] = multivariate_normal(self.mus[Ss[0]], self.Sigmas[Ss[0]])
    else:
      xs[0] = x_init
    # Perform time updates
    for t in range(0,T-1):
      s = Ss[t]
      A = self.As[s]
      b = self.bs[s]
      Q = self.Qs[s]
      xs[t+1] = multivariate_normal(dot(A,xs[t]) + b, Q)
      sample = multinomial(1,self.Z[s],size=1)[0]
      Ss[t+1] = nonzero(sample)[0][0]
    return (xs, Ss)

  def Viterbi(self, xs):
    """
    Identify the most likely hidden state sequence.
    """
    K, mus, Sigmas,Z = self.K, self.mus, self.Sigmas, self.Z
    (T, xdim) = shape(xs)
    logVs = zeros((T, K))
    Ptr = zeros((T,K))
    assignment = zeros(T)
    for k in range(K):
      logVs[0,k] = log_multivariate_normal_pdf(xs[0], mus[k], Sigmas[k])
      Ptr[0,k] = k
    for t in range(T-1):
      for k in range(K):
        logVs[t+1,k] = log_multivariate_normal_pdf(xs[t+1], mus[k],
            Sigmas[k]) + max(log(Z[:,k]) + logVs[t])
        Ptr[t+1,k] = argmax(log(Z[:,k]) + logVs[t])
    assignment[T-1] = argmax(logVs[T-1,k])
    for t in range(T-1,0,-1):
      assignment[t-1] = Ptr[t, assignment[t]]
    return assignment

  def em(self, ys, em_iters=10, em_vars='all'):
    """
    Expectation Maximization
    Inputs:
      ys: A sequence of shape (T, y_dim)
    Outputs:
      None (all updates are made to internal states)
    """
    (T,_) = shape(ys)
    K, x_dim, y_dim = self.K, self.x_dim, self.y_dim

    # regularization
    alpha = 0.1
    itr = 0
    W_i_Ts = zeros((em_iters, T, K))
    assignments = self.Viterbi(ys)
    while itr < em_iters:
      #print "\t\tLog_ll = %s" % str(log_ll_ts[T-1])
      assignments = self.Viterbi(ys)
      W_i_T = assignment_to_weights(assignments, K)
      Zhat = transition_counts(assignments, K)
      M_tt_1T = tile(Zhat, (T, 1, 1))
      self.em_update(W_i_T, M_tt_1T,
          ys, ys, alpha, itr, em_vars)
      W_i_Ts[itr] = W_i_T
      itr += 1

  def compute_sufficient_statistics(self, W_i_T, M_tt_1T, xs):
    (T, x_dim) = shape(xs)
    K = self.K

    stats = {}
    stats['cor'] = zeros((K, x_dim, x_dim))
    stats['cov'] = zeros((K, x_dim, x_dim))
    stats['cov_but_last'] = zeros((K, x_dim, x_dim))
    stats['mean'] = zeros((K, x_dim))
    stats['mean_but_first'] = zeros((K, x_dim))
    stats['mean_but_last'] = zeros((K, x_dim))
    stats['transitions'] = zeros((K,K))
    # Use Laplace Pseudocounts
    stats['total'] = ones(K)
    stats['total_but_last'] = ones(K)
    stats['total_but_first'] = ones(K)
    for t in range(T):
      for k in range(K):
        if t > 0:
          stats['cor'][k] += W_i_T[t,k] * outer(xs[t], xs[t-1])
          stats['cov'][k] += W_i_T[t,k] * outer(xs[t], xs[t])
          stats['total_but_first'][k] += W_i_T[t,k]
          stats['mean_but_first'][k] += W_i_T[t,k] * xs[t]
        stats['mean'][k] += W_i_T[t,k] * xs[t]
        stats['total'][k] += W_i_T[t,k]
        if t < T:
          stats['total_but_last'][k] += W_i_T[t,k]
          stats['mean_but_last'][k] += W_i_T[t,k] * xs[t]
          stats['cov_but_last'][k] += W_i_T[t,k] * outer(xs[t], xs[t])
    stats['transitions'] = M_tt_1T[0]
    return stats

  def em_update(self, W_i_T, M_tt_1T, xs, ys, alpha, itr,
      em_vars='all'):
    """
    Inputs:
      W_i_T= P[S_{t}=i|x_{1:T}]
      M_tt_1Ts = \sum_t P[S_t=j,S_{t+1}=k|x_{1:T}]
      em_vars: Variables to learn
    """
    K, x_dim = self.K, self.x_dim
    (T, _) = shape(xs)
    P_cur_prev = zeros((T,x_dim,x_dim))
    P_cur = zeros((T, x_dim, x_dim))
    means, covars = empirical_wells(xs, W_i_T)
    stats = self.compute_sufficient_statistics(W_i_T, M_tt_1T, xs)
    for t in range(T):
      if t > 0:
        P_cur_prev[t] = outer(xs[t], xs[t-1])
      P_cur[t] = outer(xs[t], xs[t])
    # Update As
    if 'As' in em_vars:
      self.A_update(T, x_dim, W_i_T, ys, alpha, covars,
            P_cur_prev, P_cur, itr)
    # Update bs
    if 'bs' in em_vars:
      self.b_update(T, x_dim, W_i_T, ys, alpha, means,
          covars, P_cur_prev, P_cur, itr)
    # Update Qs
    if 'Qs' in em_vars:
      self.Q_update(stats, T, x_dim, W_i_T, ys, alpha, itr, covars)
    # Update mus
    if 'mus' in em_vars:
      self.mu_update(stats, W_i_T, ys, itr)
    # Update Z
    if 'Z' in em_vars:
      self.Z_update(stats, W_i_T, M_tt_1T, T, K)

  def b_update(self, T, x_dim, W_i_T, ys, alpha,
      means, covars, P_cur_prev, P_cur, itr):
    K, x_dim = self.K, self.x_dim
    for i in range(K):
      self.bs[i] = dot(eye(x_dim) - self.As[i], means[i])

  def mu_update(self, stats, W_i_T, xs, itr):
    (T, x_dim) = shape(xs)
    K, x_dim = self.K, self.x_dim
    for k in range(K):
      # Use Laplace Pseudocount
      self.mus[k] = stats['mean'][k] / (stats['total'][k])

  def Z_update(self, stats, W_i_T, M_tt_1T, T, K):
    K = self.K
    Z = zeros((K,K))
    for i in range(K):
      for j in range(K):
        Z[i,j] += (T-1) * stats['transitions'][i,j]
        #for t in range(1,T):
        #  Z_denom += W_i_T[t,i]
        Z_denom = stats['total_but_last'][i]
        Z[i,j] /= Z_denom
    for i in range(K):
      s = sum(Z[i,:])
      Z[i,:] /= s
    self.Z = Z

  def A_update(self, T, x_dim, W_i_T, ys, alpha, covars,
        P_cur_prev, P_cur, itr):
    eta = 0.5
    #eta = 0.99
    for i in range(self.K):
      Anum = zeros((x_dim, x_dim))
      Adenom = zeros((x_dim, x_dim))
      for t in range(1,T):
        Anum += W_i_T[t,i] * P_cur_prev[t]
        Adenom += W_i_T[t,i] * P_cur[t-1]
      self.As[i] = dot(Anum, linalg.pinv(Adenom))
      u,s,v = svd(self.As[i])
      s = maximum(minimum(s,ones(shape(s))), -1.0 * ones(shape(s)))
      self.As[i] = eta * dot(u,dot(diag(s), v))

  def Q_update(self, stats, T, x_dim, W_i_T, xs, alpha, itr, covars):
    eta = 2.0/(itr+2)
    Lambda = alpha * eye(x_dim)
    for i in range(self.K):
      A = self.As[i]
      b = reshape(self.bs[i], (x_dim, 1))
      Qnum = ((stats['cov'][i]
               - dot(stats['cor'][i], A.T)
               - dot(stats['mean_but_first'][i], b.T)) +
              (-dot(A,stats['cor'][i].T) +
                dot(A,dot(stats['cov_but_last'][i], A.T)) +
                dot(A,dot(stats['mean_but_last'][i], b.T))) +
              (-dot(b,stats['mean_but_first'][i].T) +
                dot(b,dot(stats['mean_but_last'][i].T, A.T)) +
                stats['total_but_first'][i] * dot(b, b.T)))
      Qdenom = stats['total_but_first'][i]
      self.Qs[i] = (1.0/Qdenom) * Qnum
      #Qdenom = alpha
      #Qnum = Lambda
      #Anum = zeros((x_dim, x_dim))
      #Adenom = zeros((x_dim, x_dim))
      #for t in range(1,T):
      #  x_t = xs[t]
      #  x_t_pred = dot(self.As[i], xs[t-1]) + self.bs[i]
      #  diff = x_t - x_t_pred
      #  Qnum += W_i_T[t,i] * outer(diff, diff)
      #  Qdenom += W_i_T[t,i]
      #Qgrad = -0.5 * Qnum + 0.5 * Qdenom * self.Qs[i]
      #Qpred = (1.0/Qdenom) * Qnum
      #self.Qs[i] = Qpred

  def compute_metastable_wells(self):
    """Compute the metastable wells according to the formula
        x_i = (I - A)^{-1}b
      Output: wells
    """
    K, x_dim = self.K, self.x_dim
    wells = zeros((K, x_dim))
    for i in range(K):
      wells[i] = dot(inv(eye(x_dim) - self.As[i]), self.bs[i])
    return wells

  def compute_process_covariances(self):
    K, x_dim = self.K, self.x_dim
    covs = zeros((K,x_dim, x_dim))
    N = 10000
    for k in range(K):
      A = self.As[k]
      Q = self.Qs[k]
      V = iter_vars(A,Q,N)
      covs[k] = V
    return covs

  def compute_eigenspectra(self):
    K, x_dim = self.K, self.x_dim
    eigenspectra = zeros((K, x_dim,x_dim))
    for k in range(K):
      eigenspectra[k] = diag(eig(self.As[k])[0])
    return eigenspectra
