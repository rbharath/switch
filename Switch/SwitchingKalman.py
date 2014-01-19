import numpy as np
from numpy import dot, shape, eye, outer, sqrt, sum, log, diagonal, zeros
from numpy import nonzero, exp, shape, reshape, diag, maximum, isnan
from numpy import copy, minimum, ones, newaxis, mod, Inf
from numpy.linalg import svd, inv
from numpy.random import randint, randn, multinomial, multivariate_normal
from numpy.random import rand
from Kmeans import *
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

EPSILON=1e-15
GAMMA = 0.0
DISP_TIME = 1000000
# Regularization

class SwitchingKalmanFilter(object):
  """Implements a Switching Kalman Filter along with an EM algorithm.
  """
  def __init__(self, x_dim, y_dim, K=None, As=None, bs=None, Qs=None,
      Cs=None, Rs=None, Z=None, pi=None, mus=None,
      Sigmas=None):
    """
    Inputs:
      x_dim: dimension of hidden state
      y_dim: dimension of observable state
      K: Number of switching states.
      As: System transition matrices for each switching state.
      Qs: System covariance matrices for each switching state.
      Cs: Observation matrices for each switching state.
      Rs: Observation covariance matrices for each switching state.
      Z: The switching matrix for the discrete state
      pi: The initial distribution over the discrete states
      mus: the initial means for each kalman filter
      Sigmas: the initial covariances for each kalman filter
      dim_state: dimension of hidden linear dynamical systems.
      dim_obs: dimension of observation.
    """
    self.x_dim = x_dim
    self.y_dim = y_dim
    if K != None:
      self.K = K
    else:
      self.K = 1
    K = self.K
    if As != None and shape(As) == (K, x_dim, x_dim):
      self.As = copy(As)
    else:
      self.As = zeros((K,x_dim,x_dim))
      for i in range(K):
        A = randn(x_dim, x_dim)
        # Stabilize A
        u,s,v = svd(A)
        s = minimum(s,1.0)
        self.As[i] = dot(u, dot(diag(s), v))
    if bs != None and shape(bs) == (K, x_dim):
      self.bs = copy(bs)
    else:
      # Need to fix this for learning
      #self.bs = zeros((K,x_dim))
      self.bs = randn(K,x_dim)
    if Qs != None and shape(Qs) == (K, x_dim, x_dim):
      self.Qs = copy(Qs)
    else:
      self.Qs = zeros((K,x_dim, x_dim))
      for i in range(K):
        r = rand(x_dim, x_dim)
        r = (1.0/x_dim) * dot(r.T,r)
        self.Qs[i] = r
    if Cs != None and shape(Cs) == (K, y_dim, x_dim):
      self.Cs = copy(Cs)
    else:
      self.Cs = zeros((K, y_dim, x_dim))
      for i in range(K):
        if y_dim >= x_dim:
          C = rand(x_dim, x_dim)#eye(x_dim)
          # Stabilize C
          u,s,v = svd(C)
          s = minimum(s,1.0)
          C = dot(u, dot(diag(s), v))
          self.Cs[i,:x_dim,:] = C
        else:
          C = rand(y_dim, y_dim)#eye(y_dim)
          # Stabilize C
          u,s,v = svd(C)
          s = minimum(s,1.0)
          C = dot(u, dot(diag(s), v))
          self.Cs[i,:,:y_dim] = C
    if Rs != None and shape(Rs) == (K, y_dim, y_dim):
      self.Rs = copy(Rs)
    else:
      self.Rs = zeros((K,y_dim,y_dim))
      for i in range(K):
        self.Rs[i] = 0.01 * eye(y_dim)
    if Z != None and shape(Z) == (K,K):
      self.Z = copy(Z)
    else:
      self.Z = rand(K,K)
      self.Z = self.Z / (sum(self.Z,axis=0))
    if pi != None and shape(pi) == (K,):
      self.pi = copy(pi)
    else:
      # Choose random initial distribution
      vec = rand(K)
      self.pi = vec/sum(vec)
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

  def reinitialize_params(self, em_vars):
    K = self.K
    x_dim = self.x_dim
    if 'bs' in em_vars:
      for i in range(K):
        b = rand(x_dim)
        self.bs[i] = b
    if 'As' in em_vars:
      for i in range(K):
        A = randn(x_dim, x_dim)
        # Stabilize A
        u,s,v = svd(A)
        s = maximum(minimum(s,1.0), -1.0)
        self.As[i] = dot(u, dot(diag(s), v))
    if 'Qs' in em_vars:
      for i in range(K):
        r = rand(x_dim, x_dim)
        r = dot(r,r.T)
        self.Qs[i] = 0.01 * r
    if 'Cs' in em_vars:
      for i in range(K):
        if y_dim >= x_dim:
          self.Cs[i,:x_dim,:] = eye(x_dim)
        else:
          self.Cs[i,:,:y_dim] = eye(y_dim)
    if 'Rs' in em_vars:
      for i in range(K):
        self.Rs[i] = 0.01 * eye(y_dim)
    if 'Z' in em_vars:
      self.Z = rand(K,K)
      self.Z = self.Z / (sum(self.Z,axis=0))
    if 'pi' in em_vars:
      self.pi = rand(K)
      self.pi = self.pi/sum(self.pi)
    if 'mus' in em_vars:
      self.mus = zeros((K,x_dim))
      for i in range(K):
        self.mus[i] = randn(x_dim)
    if 'Sigmas' in em_vars:
      for i in range(K):
        r = rand(x_dim, x_dim)
        r = dot(r,r.T)
        self.Sigmas[i] = 0.1 * eye(x_dim) + r

  def simulate(self, T, s_init=None,x_init=None,y_init=None):
    """
    Inputs:
      T: time to run simulation
    Outputs:
      xs: Hidden continuous states
      Ss: Hidden switch states
      ys: observation states
    """
    x_dim, y_dim, = self.x_dim, self.y_dim
    # Allocate Memory
    xs = zeros((T, x_dim))
    Ss = zeros(T)
    ys = zeros((T, y_dim))
    # Sample Start conditions
    sample = multinomial(1,self.pi, size=1)
    if s_init == None:
      Ss[0] = nonzero(sample)[0][0]
    else:
      Ss[0] = s_init
    if x_init == None:
      xs[0] = multivariate_normal(self.mus[Ss[0]], self.Sigmas[Ss[0]])
    else:
      xs[0] = x_init
    if y_init == None:
      ys[0] = multivariate_normal(dot(self.Cs[Ss[0]],xs[0]),self.Rs[Ss[0]])
    else:
      ys[0] = y_init
    # Perform time updates
    for t in range(0,T-1):
      s = Ss[t]
      A = self.As[s]
      b = self.bs[s]
      Q = self.Qs[s]
      C = self.Cs[s]
      R = self.Rs[s]
      xs[t+1] = multivariate_normal(dot(A,xs[t]) + b, Q)
      ys[t+1] = multivariate_normal(dot(C,xs[t+1]), R)
      sample = multinomial(1,self.Z[s],size=1)[0]
      Ss[t+1] = nonzero(sample)[0][0]
    return (xs, Ss, ys)

  def filter(self, ys):
    """
    Inputs:
      ys: Evidence matrix of size (T,y_dim)
    Outputs:
      logM_t_1t = log P[S_{t-1}=i,S_t = j|y_{1:t}]
      M_tt = P[S_t = j|y_{1:t}]
      W_ij = P[S_{t-1}=i|S_t=j,y_{1:t}]
      x_j_tt = E[X_t|y_{1:t},S_t = j]
      V_j_tt = Cov[X_t|y_{1:t}, S_t = j]
      V_ij_tt_1 = Cov[X_t,X_{t-1}|y_{1:t},S_{t-1}=i,S_t=j]
      x_tt = E[X_t|y_{1:t}]
    Intermediate:
      L_ij_t = P[y_t|y_{1:t-1},S_{t-1} = i, S_t = j]
    """
    DEBUG = True
    T = shape(ys)[0]
    x_dim, K = self.x_dim, self.K
    # Allocate Memory for Outputs
    logM_t_1t = zeros((T,K,K))
    logM_tt = zeros((T,K))
    W_ij = zeros((T,K,K))
    F_ij = zeros((T,K,K))
    x_j_tt = zeros((T,K,x_dim))
    x_tt = zeros((T,x_dim))
    V_j_tt = zeros((T,K,x_dim,x_dim))
    log_ll_j_t = zeros((T,K))
    log_ll_t = zeros(T)
    # Allocate Memory for intermediate matrices
    x_ij_tt = zeros((T,K,K,x_dim))
    V_ij_tt = zeros((T,K,K,x_dim,x_dim))
    V_ij_tt_1 = zeros((T,K,K,x_dim,x_dim))
    L_ij_t = zeros((T,K,K))
    # Initial Conditions
    for j in range(K):
      (x_j_tt[0,j], V_j_tt[0,j],_,L_j) = Filter(self.mus[j],
          self.Sigmas[j], ys[0], eye(x_dim), zeros(x_dim),
          zeros((x_dim,x_dim)), self.Cs[j], self.Rs[j])
      log_ll_j_t[0,j] = L_j
    logM_tt[0] = log(self.pi)
    for j in range(K):
      log_ll_t[0] += exp(logM_tt[0,j]) * log_ll_j_t[0,j]
    # Perform Forward Update Loop
    for t in range(1,T):
      for i in range(K):
        for j in range(K):
          (x_ij_tt[t,i,j], V_ij_tt[t,i,j], V_ij_tt_1[t,i,j],
              L_ij_t[t,i,j]) = Filter(x_j_tt[t-1,i],
                  V_j_tt[t-1,i], ys[t], self.As[j], self.bs[j],
                  self.Qs[j], self.Cs[j], self.Rs[j])
      for i in range(K):
        for j in range(K):
          logM_t_1t[t,i,j] = (L_ij_t[t,i,j] + log(self.Z[i,j])
              + logM_tt[t-1,i])
      # sum out the last two dimensions
      logS = logsumexp(logsumexp(logM_t_1t[t]))
      logM_t_1t[t] = logM_t_1t[t] - logS
      logM_tt[t] = logsumexp(logM_t_1t[t], dim=0)
      for j in range(self.K):
        W_ij[t,:,j] = logM_t_1t[t,:,j] - logM_tt[t,j]
        W_ij[t,:,j] -= logsumexp(W_ij[t,:,j])
        W_ij[t,:,j] = exp(W_ij[t,:,j])
        #else:
        #  W_ij[t,:,j] = 1 # set to uniform in absence of other info
        #if sum(W_ij[t,:,j]) == 0:
        #  raise ValueError
        #if M_tt[t-1,j] > 0:
        F_ij[t,j,:] = logM_t_1t[t,j,:] - logM_tt[t-1,j]
        F_ij[t,j,:] -= logsumexp(F_ij[t,j,:])
        F_ij[t,j,:] = exp(F_ij[t,j,:])
        #if sum(F_ij[t,j,:]) == 0:
        #  raise ValueError
      for i in range(K):
        #if M_tt[t-1,i] > 0:
        for j in range(K):
          #if F_ij[t,i,j] > 0:
          log_ll_j_t[t,j] += F_ij[t,i,j] * (log_ll_j_t[t-1,i] +
              L_ij_t[t,i,j])
      for j in range(K):
        log_ll_t[t] += exp(logM_tt[t,j]) * log_ll_j_t[t,j]
      for j in range(self.K):
        (x_j_tt[t,j], _, V_j_tt[t,j]) =\
            Collapse(x_ij_tt[t,:,j], V_ij_tt[t,:,j], W_ij[t,:,j])
    # Compute Average Filter Estimate
    for k in range(K):
      x_tt += exp(logM_tt[:,k:k+1]) * x_j_tt[:,k]
    return (logM_t_1t, logM_tt, W_ij, x_j_tt, x_tt, V_j_tt, V_ij_tt_1,
        log_ll_j_t, log_ll_t)

  def smooth(self, x_j_tt, V_j_tt, V_ij_tt_1, logM_tt, ys):
    """
    Inputs:
      x_j_tt = E[X_t|S_t=j,y_{1:t}]
      V_j_tt = Cov[X_t|S_t=j,y_{1:t}]
      V_ij_tt_1 = Cov[X_t,X_{t-1}|S_{t-1}=j,S_j=j,y_{1:t}]
      logM_tt = log P[S_t = j|y_{1:t}]
      ys: observations
    Outputs:
      x_j_tT = E[X_t|S_t=j,y_{1:T}]
      V_j_tT = Cov[X_t|S_t=j,y_{1:T}]
      logU_jk_t = P[S_t=j|S_{t+1}=k,y_{1:T}]
      logM_tt_1T = log P[S_t=j,S_{t+1}=k|y_{1:T}]
      logM_tT = log P[S_t=j|y_{1:T}]
      W_kj_tT = P[S_{t+1}=k|S_t=j,y_{1:T}]
      x_tT = E[X_t|y_{1:T}]
      V_tT = Cov[X_t|y_{1:T}]
    Intermediates:
      x_jk_tT = E[x_{t}|y_{1:T}, S_{t}=j, S_{t+1}=k] \approx x_j_tT
      x_j_forwardk_tT = E[x_{t}|y_{1:T},S_{t}=k,S_{t-1}=j]
      V_jk_tT = Cov[X_t|y_{1:T}, S_{t}=j, S_{t+1}=k]
      V_jk_tt_1T = Cov[X_t, X_{t-1}|y_{1:T}, S_{t}=j, S_{t+1}=k]
      V_j_tt_1T = Cov[X_{t},X_{t-1}|S_{t}=j,y_{1:T}]
      x_e_k_tT = E[X_t|y_{1:T},S_{t+1}=k]
      V_tt_1T = E[X_t,X_{t-1}|y_{1:T}]
    """
    DEBUG = False
    T = shape(ys)[0]
    # Extract Dimensionality Information
    x_dim, K = self.x_dim, self.K
    # Allocate Output Memory
    x_j_tT = zeros((T,K,x_dim))
    V_j_tT = zeros((T,K,x_dim, x_dim))
    logU_jk_t = zeros((T,K,K))
    logM_tt_1T = zeros((T,K,K))
    logM_tT = zeros((T,K))
    logW_kj_tT = zeros((T,K,K))
    x_j_tT = zeros((T,K,x_dim))
    V_j_tT = zeros((T,K,x_dim,x_dim))
    x_tT = zeros((T,x_dim))
    V_tT = zeros((T,x_dim,x_dim))
    # Allocate Intermediate Memory
    x_jk_tT = zeros((T,K,K,x_dim))
    x_j_forwardk_tT = zeros((T,K,K,x_dim))
    V_jk_tT = zeros((T,K,K,x_dim, x_dim))
    V_jk_tt_1T = zeros((T,K,K,x_dim, x_dim))
    V_j_tt_1T = zeros((T,K,x_dim,x_dim))
    x_e_k_tT = zeros((T,K,x_dim))
    V_tt_1T = zeros((T,x_dim,x_dim))
    # Allocate Initial Conditions
    x_j_tT[T-1] = x_j_tt[T-1]
    V_j_tT[T-1] = V_j_tt[T-1]
    logM_tT[T-1] = logM_tt[T-1]
    (x_tT[T-1], _, V_tT[T-1]) = \
        Collapse(x_j_tT[T-1], V_j_tT[T-1], exp(logM_tT[T-1]))
    # Perform Backward Update Loop
    for t in range(T-2,-1,-1):
      for j in range(K):
        for k in range(K):
          (x_jk_tT[t,j,k], V_jk_tT[t,j,k], V_jk_tt_1T[t+1,j,k]) =\
            Smooth(x_j_tT[t+1,k], V_j_tT[t+1,k],
                    x_j_tt[t,j], V_j_tt[t,j], V_j_tt[t+1,k],
                    V_ij_tt_1[t+1,j,k], self.As[k], self.bs[k], self.Qs[k])
      for j in range(K):
        for k in range(K):
          logU_jk_t[t, j, k] = logM_tt[t+1,j] + log(self.Z[j,k])
      logU_jk_t[t] = logU_jk_t[t] - logsumexp(logU_jk_t[t], dim=0)
      for j in range(K):
        for k in range(K):
          logM_tt_1T[t,j,k] = logU_jk_t[t,j,k] + logM_tT[t+1,k]
      for j in range(K):
        logM_tT[t,j] = logsumexp(logM_tt_1T[t,j])
      for j in range(K):
        for k in range(K):
          logW_kj_tT[t,k,j] = logM_tt_1T[t,j,k] - logM_tT[t,j]
      for j in range(K):
        (x_j_tT[t,j], _, V_j_tT[t,j]) =\
            Collapse(x_jk_tT[t,j], V_jk_tT[t,j], exp(logW_kj_tT[t, :, j]))
      (x_tT[t], _, V_tT[t]) = \
          Collapse(x_j_tT[t], V_j_tT[t], exp(logM_tT[t]))
      for j in range(K):
        x_j_forwardk_tT[t+1,:,k] = x_j_tT[t+1,k] # An Approximation
        for k in range(K):
          (_,_,V_j_tt_1T[t+1,k]) = CollapseCross(
              x_j_forwardk_tT[t+1,:,k],
              x_jk_tT[t,:,k], V_jk_tt_1T[t+1,:,k], exp(logU_jk_t[t,:,k]))
      for k in range(K):
        s = zeros(x_dim)
        for j in range(K):
          s += x_jk_tT[t,j,k] * exp(logU_jk_t[t,j,k])
        x_e_k_tT[t,k] = s
      (_,_,V_tt_1T[t+1]) = CollapseCross(x_j_tT[t+1], x_e_k_tT[t],
                              V_j_tt_1T[t+1], exp(logM_tT[t+1]))
    return (x_j_tT, V_j_tT, logU_jk_t, logM_tt_1T, logM_tT, logW_kj_tT, x_tT, V_tT)

  def inference(self, xs):
    """
    Inputs:
      xs: E[x_t | y_{1:T}]
    Outputs:
      S_j_t_x_1t: P[S_t = j|x_{1:t}] (forward pass)
      S_j_t_x_1T: P[S_t = j|x_{1:T}] (backward pass)
      S_jk_tt_1_x_1T: P[S_t=j,S_{t+1}=k|x_{1:T}] (joint)
    Intermediates:
      S_k_t_1_x_1t = P[S_{t+1} = k|x_{1:t}]
    """
    DEBUG = False
    (T,x_dim) = shape(xs)
    K = self.K
    # Initialize Memory
    S_j_t_x_1t = zeros((T,K))
    S_k_t_1_x_1t = zeros((T,K))
    S_j_t_x_1T = zeros((T,K))
    S_jk_tt_1_x_1T = zeros((T,K,K))
    # forward passs
    for t in range(T):
      for j in range(K):
        if t > 0:
          L_t_j = exp(log_multivariate_normal_pdf(xs[t],
                        dot(self.As[j], xs[t-1]),self.Qs[j]))
        else:
          L_t_j = exp(log_multivariate_normal_pdf(xs[t],
                        self.mus[j], self.Sigmas[j]))
        s = 0
        if L_t_j == 0:
          L_t_j = EPSILON
        for i in range(K):
          if t > 0:
            s += self.Z[i,j] * S_j_t_x_1t[t-1,i]
          else:
            s += self.Z[i,j] * self.pi[i]
        s *= L_t_j
        S_j_t_x_1t[t,j] = s
      c = sum(S_j_t_x_1t[t])
      if c == 0:
        raise ValueError
      S_j_t_x_1t[t] /= c
    # Compute the one-step look-ahead matrix
    for t in range(T-1):
      # We shouldn't need a normalization step afterwards, but should
      # check anyways to be sure
      for k in range(K):
        s = 0
        for j in range(K):
          s += S_j_t_x_1t[t,j] * self.Z[j,k]
        S_k_t_1_x_1t[t,k] = s
    # Initialize the backwards pass
    S_j_t_x_1T[T-1] = S_j_t_x_1t[T-1]
    # backward pass
    for t in range(T-2,-1,-1):
      for j in range(K):
        s = 0
        for k in range(K):
          const_k = S_j_t_x_1t[t,j] * self.Z[j,k] / (S_k_t_1_x_1t[t,k])
          s += const_k * S_j_t_x_1T[t+1,k]
        S_j_t_x_1T[t,j] = s
    # joint pass
    for t in range(T-1):
      for j in range(K):
        for k in range(K):
         S_jk_tt_1_x_1T[t,j,k] = S_j_t_x_1T[t,j] * self.Z[j,k]
    return (S_j_t_x_1t, S_j_t_x_1T, S_jk_tt_1_x_1T)

  def fully_observable_daem(self,xs, ys, em_num_sched=10,
      em_iters=10,em_vars='all'):
    """
    Inputs:
      xs: A sequence
    Outputs:
      None (all updates are made to internal states)
    """
    if em_vars == 'all':
      em_vars = ['As', 'Qs', 'mus', 'Sigmas', 'Z', 'pi', 'Cs', 'Rs']
    (T,_) = shape(xs)
    K, x_dim = self.K, self.x_dim
    S_j_t_x_1Ts = zeros((T,K))
    S_jk_tt_1_x_1Ts = zeros((T,K,K))
    beta = 0.10/em_num_sched
    const = (1.0/beta) ** (1.0/max(em_num_sched-1,1))
    for sched in range(em_num_sched):
      print "EM Schedule Num = %d" % sched
      print "beta = %f" % beta
      print "const = %f" % const
      for itr in range(em_iters):
        print "EM Iteration = %d" % itr
        # Now perform inference
        (_, S_j_t_x_1T, S_jk_tt_1_x_1T) = self.inference(xs)
        self.em_update(S_j_t_x_1Ts, S_jk_tt_1_x_1Ts, xs, ys,
            sched, itr, em_vars, beta)
      beta = min(beta * const, 1.0)

  def daem(self, ys, em_num_sched = 10, em_iters=10, em_vars='all'):
    """
    Deterministic Annealing EM
    Inputs:
      ys: A sequence of shape (T, y_dim)
    Outputs:
      None (all updates are made to internal states)
    """
    if em_vars == 'all':
      em_vars = ['As', 'Qs', 'mus', 'Sigmas', 'Zs', 'pis', 'Cs', 'Rs']
    (T,_) = shape(ys)
    K, x_dim, y_dim = self.K, self.x_dim, self.y_dim
    # regularization
    alpha = 0.1 * T

    # For debugging purposes
    out_x_tts = zeros((em_num_sched, em_iters,T,x_dim))
    out_x_tTs = zeros((em_num_sched, em_iters,T,x_dim))
    out_x_j_tTs = zeros((em_num_sched, em_iters,T,K,x_dim))
    out_x_j_tts = zeros((em_num_sched, em_iters,T,K,x_dim))
    out_V_j_tts = zeros((em_num_sched, em_iters,T,K,x_dim,x_dim))
    out_log_ll_j_ts = zeros((em_num_sched, em_iters,T,K))
    out_log_ll_ts = zeros((em_num_sched, em_iters,T))
    out_W_i_Ts = zeros((em_num_sched, em_iters,T,K))
    mus_iter = zeros((em_num_sched, em_iters, K, x_dim))
    Sigmas_iter = zeros((em_num_sched, em_iters, K, x_dim, x_dim))
    out_Ds = zeros((em_num_sched, em_iters, K, x_dim, x_dim))
    As_iter = zeros((em_num_sched, em_iters, K, x_dim, x_dim))
    bs_iter = zeros((em_num_sched, em_iters, K, x_dim))
    Qs_iter = zeros((em_num_sched, em_iters, K, x_dim, x_dim))
    Cs_iter = zeros((em_num_sched, em_iters, K, y_dim, x_dim))
    Rs_iter = zeros((em_num_sched, em_iters, K, y_dim, y_dim))
    Z_iter = zeros((em_num_sched, em_iters, K, K))
    pi_iter = zeros((em_num_sched, em_iters, K))
    beta = 1.0/em_num_sched
    const = (1.0/beta) ** (1.0/max(em_num_sched-1,1))
    for sched in range(em_num_sched):
      print "EM Schedule Num = %d" % sched
      print "beta = %f" % beta
      print "const = %f" % const
      itr = 0
      while itr < em_iters:
        mus_iter[sched, itr] = self.mus
        Sigmas_iter[sched, itr] = self.Sigmas
        As_iter[sched, itr] = self.As
        Qs_iter[sched, itr] = self.Qs
        Cs_iter[sched, itr] = self.Cs
        Rs_iter[sched, itr] = self.Rs
        pi_iter[sched, itr] = self.pi
        Z_iter[sched, itr] = self.Z
        bs_iter[sched, itr] = self.bs
        #try:
        print "\tEM Iteration = %d" % itr
        (_, logM_tts, _, x_j_tts, x_tts,
            V_j_tts, V_ij_tt_1s, log_ll_j_ts, log_ll_ts) \
                = self.filter(ys)
        print "\t\tLog_ll = %s" % str(log_ll_ts[T-1])
        out_x_tts[sched, itr] = x_tts
        out_x_j_tts[sched, itr] = x_j_tts
        out_V_j_tts[sched, itr] = V_j_tts
        out_log_ll_j_ts[sched, itr] = log_ll_j_ts
        out_log_ll_ts[sched, itr] = log_ll_ts
        # Now perform smoothing
        (x_j_tTs, _, _, logM_tt_1Ts, _, _, x_tTs, _) = \
            self.smooth(x_j_tts, V_j_tts,
                V_ij_tt_1s, logM_tts, ys)
        out_x_tTs[sched, itr] = x_tTs
        out_x_j_tTs[sched, itr] = x_j_tTs
        # Now perform inference
        (_, S_j_t_x_1Ts, S_jk_tt_1_x_1Ts) = \
            self.inference(x_tTs)
        out_W_i_Ts[sched, itr] = S_j_t_x_1Ts
        out_Ds[sched, itr] = empirical_wells(ys,
            out_W_i_Ts[sched, itr])[1]
        self.em_update(S_j_t_x_1Ts, S_jk_tt_1_x_1Ts,
            x_tTs, ys, alpha, sched, itr, em_vars, beta)
        itr += 1
        #except FloatingPointError:
        #  itr += 1
        #  continue
      beta = min(beta * const, 1.0)
    return (out_x_tTs, out_x_tts, out_x_j_tTs, out_x_j_tts, out_V_j_tts,
        out_log_ll_j_ts, out_log_ll_ts, out_W_i_Ts, out_Ds, mus_iter,
        Sigmas_iter, As_iter, bs_iter, Qs_iter, Cs_iter, Rs_iter, pi_iter,
        Z_iter)

  def em_update(self, W_i_T, M_tt_1T, x_tT, ys, alpha, sched, itr,
      em_vars='all', beta=1.0):
    """
    TODO: Add support for C and R learning
    Inputs:
      W_i_T: The sequence satisfies
        W_i_T = P[S_{t}=i|x_{1:T}], (T, K) matrix of weights
        (is output S_j_t_x_1T of the inference method)
      M_tt_1Ts: The sequence satisfies
        M_tt_1T = P[S_t=j,S_{t+1}=k|y_{1:T}] or
        M_tt_1T = P[S_t=j,S_{t+1}=k|x_{1:T}]
        depending on whether system is fully observable.
      x_tTs: The sequence satisfies
        x_tT = E[X_t|y_{1:T}] (T, x_dim)
      em_vars: Variables to perform learning on
      beta: annealing parameter
    Outputs:
      None (all updates are made to internal state)
    """
    K, x_dim, y_dim = self.K, self.x_dim, self.y_dim
    (T, _) = shape(x_tT)
    P_cur_prev = zeros((T,x_dim,x_dim))
    P_cur = zeros((T, x_dim, x_dim))
    means, covars = empirical_wells(ys, W_i_T)
    W_i_T = W_i_T ** beta
    for t in range(T):
      if t > 0:
        P_cur_prev[t] = outer(x_tT[t], x_tT[t-1])
      P_cur[t] = outer(x_tT[t], x_tT[t])
    # Update Cs
    if 'Cs' in em_vars:
      for i in range(K):
        Cnum = zeros((y_dim, x_dim))
        Cdenom = zeros((x_dim, x_dim))
        for t in range(0,T):
          Cnum += W_i_T[t,i] * outer(ys[t], x_tT[t])
          Cdenom += W_i_T[t,i] * P_cur[t]
        C = dot(Cnum, linalg.pinv(Cdenom))
        self.Cs[i] = C
        #if x_dim == y_dim: # Figure out a way to generalize...
        #  # Stabilize C
        #  u,s,v = svd(C)
        #  s = maximum(minimum(s,1.0), -1.0)
        #  #s = minimum(s,1.0)
        #  self.Cs[i] = dot(u, dot(diag(s), v))
        #else:
        #  self.Cs[i] = C
    # Update Rs
    if 'Rs' in em_vars:
      for i in range(self.K):
        Rdenom = alpha
        if x_dim == y_dim:
          Lambda = alpha * eye(x_dim)
        R1 = zeros((y_dim, y_dim))
        R2 = zeros((x_dim, y_dim))
        for t in range(T):
          R1 += W_i_T[t,i] * outer(ys[t], ys[t])
          R2 += W_i_T[t,i] * outer(x_tT[t], ys[t])
          Rdenom += W_i_T[t,i]
        Rnum = Lambda + R1 + dot(self.Cs[i], R2)
        self.Rs[i] = (1/Rdenom) * Rnum
    # Update As
    if 'As' in em_vars:
      self.A_update(T, x_dim, W_i_T, M_tt_1T, x_tT, ys, alpha, covars,
            P_cur_prev, P_cur, itr)
    # Update bs
    if 'bs' in em_vars:
      #Lambda = alpha * ones(x_dim)
      for i in range(K):
        bnum = zeros((x_dim))
        bdenom = 0
        for t in range(1,T):
          bnum += W_i_T[t,i] * (x_tT[t] - dot(self.As[i], x_tT[t-1]))
          bdenom += W_i_T[t,i]
        self.bs[i] = (1/bdenom) * bnum
    # Update Qs
    if 'Qs' in em_vars:
      self.Q_update(T, x_dim, W_i_T, M_tt_1T, x_tT, ys, alpha, covars, itr)
    # Update mus
    if 'mus' in em_vars:
      for i in range(self.K):
        mu_num = zeros(x_dim)
        mu_denom = 0
        mu_num += W_i_T[0,i] * x_tT[0]
        mu_denom += W_i_T[0,i]
        if mu_denom > 0:
          self.mus[i] = (1/mu_denom) * mu_num
    # Update Sigmas
    if 'Sigmas' in em_vars:
      Lambda = alpha * eye(x_dim)
      for i in range(self.K):
        Sigma_num_1 = zeros((x_dim, x_dim))
        Sigma_num_2 = zeros((1, x_dim))
        Sigma_num_3 = zeros((x_dim, 1))
        Sigma_num_4 = 0
        Sigma_num_1 += W_i_T[0,i] * outer(x_tT[0], x_tT[0])
        Sigma_num_2 += W_i_T[0,i] * x_tT[0]
        Sigma_num_2 += W_i_T[0,i] * x_tT[0]
        Sigma_num_4 += W_i_T[0,i]
        Sigma_denom = Sigma_num_4
        Sigma_num = (Sigma_num_1 - outer(self.mus[i], Sigma_num_2) -
            outer(Sigma_num_3, self.mus[i].T) +
            Sigma_num_4 * outer(self.mus[i],self.mus[i]))
        prop_sigma = (1/(Sigma_denom+alpha)) * (Lambda + Sigma_num)
        # Adding in this test to ensure that we don't get non-psd matrices
        if min(linalg.eig(prop_sigma)[0]) > 0:
          self.Sigmas[i] = prop_sigma
    # Update Z
    if 'Z' in em_vars:
      Z = zeros((K,K))
      for i in range(K):
        for j in range(K):
          Z_denom = 0
          for t in range(1,T):
            Z[i,j] += M_tt_1T[t-1,i,j]
            Z_denom += W_i_T[t,i]
          Z[i,j] /= Z_denom
      self.Z = (Z.T / (sum(Z,axis=1))).T
    # Update pis
    if 'pi' in em_vars:
      for i in range(self.K):
        self.pi[i] += W_i_T[0,i]
        self.pi[i] /= K
      self.pi = self.pi/(sum(self.pi))


  def A_update(self, T, x_dim, W_i_T, M_tt_1T, x_tT, ys, alpha, covars,
        P_cur_prev, P_cur, itr):
    N_steps = 1
    Aolds = self.As
    (_,_,_,_,_,_,_,_,log_ll_curs) = self.filter(ys)
    log_ll_cur = log_ll_curs[T-1]
    for step in range(N_steps):
      eta = 1.0 * (1.0/(itr*N_steps + step+1))
      for i in range(self.K):
        D = covars[i]
        Q = self.Qs[i]
        A = self.As[i]
        Anum = zeros((x_dim, x_dim))
        Adenom = zeros((x_dim, x_dim))
        for t in range(1,T):
          term = (P_cur_prev[t] - outer(self.bs[i], x_tT[t-1]))
          Anum += W_i_T[t,i] * term
          Adenom += W_i_T[t,i] * P_cur[t-1]
        Agrad = -dot(inv(self.Qs[i]), Anum) +\
            dot(inv(self.Qs[i]), dot(self.As[i], Adenom))
        Agrad += 20 * (dot(D, dot(A.T, dot(A, dot(D.T, A.T)))) +
            dot(A, dot(D.T, dot(A.T, dot(A, D)))) +
            dot(A, dot(D.T, dot(A.T, dot(A, D)))) +
            dot(A, dot(D, dot(A.T, dot(A, D.T)))) +
            2 * (dot((Q-D).T, dot(A, D.T)) +
                 dot(Q-D, dot(A, D))))
        #A = dot(Anum, linalg.pinv(Adenom))
        A = self.As[i] + eta * Agrad
        # Stabilize A
        u,s,v = svd(A)
        s = maximum(minimum(s,1.0), -1.0)
        self.As[i] = dot(u, dot(diag(s), v))
    (_,_,_,_,_,_,_,_,log_ll_news) = self.filter(ys)
    log_ll_new = log_ll_news[T-1]
    # Primitive version of line search
    if log_ll_new < log_ll_cur:
      self.As = Aolds

  def Q_update(self, T, x_dim, W_i_T, M_tt_1T, x_tT, ys, alpha,
          covars, itr):
    N_steps = 1
    Qolds = self.Qs
    try:
      (_,_,_,_,_,_,_,_,log_ll_curs) = self.filter(ys)
      log_ll_cur = log_ll_curs[T-1]
    except FloatingPointError:
      log_ll_cur = -Inf
    for step in range(N_steps):
      eta = 1.0 * (1.0/(itr*N_steps + step+1))
      Lambda = alpha * eye(x_dim)
      for i in range(self.K):
        D = covars[i]
        A = self.As[i]
        Qforced = D - dot(A, dot(D, A.T))
        #self.Qs[i] = Qforced
        #continue
        Qdenom = alpha
        Qnum = Lambda
        Anum = zeros((x_dim, x_dim))
        Adenom = zeros((x_dim, x_dim))
        for t in range(1,T):
          x_t = x_tT[t]
          x_t_pred = dot(self.As[i], x_tT[t-1]) + self.bs[i]
          diff = x_t - x_t_pred
          if mod(t,DISP_TIME) == 0:
            print "\t\t\tt = %d" % t
            print "\t\t\tx_t = %s" % (str(x_t))
            print "\t\t\tx_t_pred = %s" % (str(x_t_pred))
            print "\t\t\tdiff = %s" % str(diff)
            print "\t\t\tW_i_T[%d,%d] = %s" % (t, i, str(W_i_T[t,i]))
            print
          Qnum += W_i_T[t,i] * outer(diff, diff)
          Qdenom += W_i_T[t,i]
        Qgrad = -0.5 * Qnum + 0.5 * Qdenom * self.Qs[i]
        Q = self.Qs[i]
        Qgrad += 20 * (- 2 * dot(Q.T, dot(Q, Q.T)) \
            - 2 * dot(Q.T, dot(dot(A,dot(D,A.T))-D,Q.T)))
        inv_update = inv(self.Qs[i]) + eta * Qgrad
        #try:
        self.Qs[i] = linalg.pinv(inv_update)
        #except FloatingPointError:
        #  #self.Qs[i] = Qolds[i]
        #  self.Qs[i] = (1.0/Qdenom) * Qnum
    try:
      (_,_,_,_,_,_,_,_,log_ll_news) = self.filter(ys)
      log_ll_new = log_ll_news[T-1]
    except FloatingPointError:
      log_ll_new = -Inf
    # Primitive version of line search
    if log_ll_new < log_ll_cur:
      self.Qs = Qolds

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

def log_multivariate_normal_pdf(x, mu, Sigma):
    size = len(x)
    if size == len(mu) and (size, size) == shape(Sigma):
      det = linalg.det(Sigma)
      if det == 0:
        raise NameError("The covariance matrix can't be singular")

      try:
        log_norm_const = -0.5 * (float(size) * log(2*np.pi) + log(det))
      except FloatingPointError:
        log_norm_const = -Inf

      x_mu = x - mu
      inv = linalg.pinv(Sigma)
      log_result = -0.5 * dot(x_mu, dot(inv, x_mu.T))
      return log_norm_const + log_result
    else:
      raise NameError("The dimensions of the input don't match")

def old_log_multivariate_normal_pdf(x,mu,Sigma, min_covar=1.e-7):
  """
  Inputs:
    x: value vector
    mu: mean vector
    Sigma: covariance matrix
  Outputs:
    log_prob: log probability density
  """
  dim = x.shape[0]
  # Hack for debugging. FIX ME!
  if dim == 1:
    return stats.norm.logpdf(x,loc=mu,scale=Sigma)
  try:
    Sigma_chol = linalg.cholesky(Sigma, lower=True)
  except linalg.LinAlgError:
    Sigma_chol = linalg.cholesky(Sigma + min_covar * eye(dim), lower=True)
  Sigma_log_det = 2 * sum(log(diagonal(Sigma_chol)))
  Sigma_sol = linalg.solve_triangular(Sigma_chol, (x - mu), lower=True)
  log_prob = - .5 * (sum(Sigma_sol ** 2) + dim * log(2*np.pi) + \
                      Sigma_log_det)
  return log_prob

def Filter(x, V, y, A, b, Q, C, R):
  """
  Inputs:
    x: x_{t-1|t-1}, the input filtered state.
    V: V_{t-1|t-1}, the input covariance.
    y: y_t, the evidence.
    A: system transition
    Q: system covariance
    C: observation
    R: observation covariance
  Outputs:
    cor_x: x_{t|t}, the updated filtered state.
    cor_V: V_{t|t}, the update covariance.
    cor_V_joint:  V_{t,t-1|t}, the joint covariance.
    logL: log N(e_t; 0, S_t)
  """
  dim_state = shape(x)[0]
  # Compute predicted state
  pred_x = dot(A, x) + b
  pred_V = dot(A, dot(V, A.T)) + Q

  # Perform the following to ensure V is psd
  if min(linalg.eig(V)[0]) < 0:
    Vsym = (V + V.T)/2
    d,D = linalg.eigh(Vsym)
    d = maximum(d, zeros(shape(d)))
    V = dot(D, dot(diag(d), D.T))

  # Compute error
  e = y - dot(C,pred_x)
  S = dot(C, dot(pred_V, C.T)) + R
  #K = dot(V, dot(pred_V, linalg.pinv(S))) WRONG!
  K = dot(pred_V, dot(C.T, linalg.pinv(S)))

  # Update the estimates
  cor_x = pred_x + dot(K, e)
  cor_V = pred_V - dot(K, dot(S, K.T))
  cor_V_joint = dot(eye(dim_state) - dot(K,C), dot(A, V))
  #logL = log_multivariate_normal_pdf(e,zeros(shape(e)),S)
  logL = log_multivariate_normal_pdf(y - cor_x,zeros(shape(e)),cor_V)
  return (cor_x, cor_V, cor_V_joint, logL)

def Smooth(x_t_1T, V_t_1T, x_tt, V_tt, V_t_1t_1, V_t_1tt_1, A, b, Q):
  """
  Inputs:
    x_t_1T: x_{t+1|T}
    V_t_1T:  V_{t+1|T}
    x_tt: x_{t|t}
    V_tt: V_{t|t}
    V_t_1t_1: V_{t+1|t+1}
    V_t_1tt_1: V_{t+1,t|t+1}
    A: transition matrix
    Q: transition covariance
  Outputs:
    x_tT: x_{t:T}
    V_tT: V_{t|T}
    V_t_1tT: V_{t+1,t|T}
  """
  x_t_1t = dot(A,x_tt) + b
  V_t_1t = dot(A,dot(V_tt, A.T)) + Q

  J = dot(V_tt, dot(A.T, linalg.pinv(V_t_1t)))
  x_tT = x_tt + dot(J, x_t_1T - x_t_1t)
  V_tT = V_tt + dot(J, dot(V_t_1T - V_t_1t, J.T))
  V_t_1tT = V_t_1tt_1 + dot(V_t_1T - V_t_1t_1,
            dot(linalg.pinv(V_t_1t_1), V_t_1tt_1))
  return (x_tT, V_tT, V_t_1tT)

def CollapseCross(mu_Xs, mu_Ys, V_Xs_Ys, Ps):
  """
  Inputs:
    mu_Xs: \mu_X^j
    mu_Ys: \mu_Y^j
    V_Xs_Ys: V_{X,Y}^j
    Ps: P^j
  Outputs:
    mu_X: moment-matched X mean
    mu_Y: moment-matched Y mean
    V_XY: moment matched covariance
  """
  N = shape(mu_Xs[:,0])[0]
  mu_X = zeros(shape(mu_Xs[0,:]))
  mu_Y = zeros(shape(mu_Ys[0,:]))
  V_XY = zeros(shape(V_Xs_Ys[0,:,:]))
  for j in range(N):
    mu_X += Ps[j] * mu_Xs[j,:]
    mu_Y += Ps[j] * mu_Ys[j,:]
  for j in range(N):
    V_XY += Ps[j] * V_Xs_Ys[j,:,:]
    V_XY += Ps[j] * outer(mu_Xs[j,:] - mu_X, mu_Ys[j,:] - mu_Y)
  # Perform the following to ensure V_XY is psd
  try:
    if min(linalg.eig(V_XY)[0]) < 0:
      Vsym = (V_XY + V_XY.T)/2
      d,D = linalg.eigh(Vsym)
      d = maximum(d, zeros(shape(d)))
      V_XY = dot(D, dot(diag(d), D.T))
  except ValueError:
    V_XY = 0.1 * eye(len(mu_X))
  return (mu_X, mu_Y, V_XY)

def Collapse(mu_Xs, V_Xs, Ps):
  """
  Inputs:
    mu_Xs: \mu_X^j
    V_Xs: V_{X}^j
    Ps: P^j
  Outputs:
  """
  return CollapseCross(mu_Xs, mu_Xs, V_Xs, Ps)

def logsumexp(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.

       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is the default), logsumexp is
       computed along the last dimension.
    """
    if len(x.shape) < 2:
        xmax = x.max()
        return xmax + log(sum(exp(x-xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim+1, len(x.shape)) + [dim])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        return xmax + log(sum(exp(x-xmax[...,newaxis]),lastdim))

def iter_vars(A, Q,N):
  V = eye(shape(A)[0])
  for i in range(N):
    V = Q + dot(A,dot(V, A.T))
  return V

def empirical_wells(Ys, W_i_Ts):
  (T, y_dim) = shape(Ys)
  (_, K) = shape(W_i_Ts)
  means = zeros((K, y_dim))
  covars = zeros((K, y_dim, y_dim))
  for k in range(K):
    num = zeros(y_dim)
    denom = 0
    for t in range(T):
      num += W_i_Ts[t, k] * Ys[t]
      denom += W_i_Ts[t,k]
    means[k] = (1.0/denom) * num
  for k in range(K):
    num = zeros((y_dim, y_dim))
    denom = 0
    for t in range(T):
      num += W_i_Ts[t, k] * outer(Ys[t] - means[k], Ys[t] - means[k])
      denom += W_i_Ts[t,k]
    covars[k] = (1.0/denom) * num
  return means, covars