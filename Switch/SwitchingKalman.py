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

EPSILON=1e-15

class SwitchingKalmanFilter(object):
  """Implements a Switching Kalman Filter along with an EM algorithm.
  """
  def __init__(self, x_dim, y_dim, K=None, As=None, bs=None, Qs=None,
      Z=None, pi=None, mus=None, Sigmas=None):
    """
    Inputs:
      x_dim: dimension of hidden state
      y_dim: dimension of observable state
      K: Number of switching states.
      As: System transition matrices for each switching state.
      Qs: System covariance matrices for each switching state.
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
        F_ij[t,j,:] = logM_t_1t[t,j,:] - logM_tt[t-1,j]
        F_ij[t,j,:] -= logsumexp(F_ij[t,j,:])
        F_ij[t,j,:] = exp(F_ij[t,j,:])
      for i in range(K):
        for j in range(K):
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
    base_means = None
    means, assignments = kmeans(ys, K)
    while itr < em_iters:
      print "\tEM Iteration = %d" % itr
      (_, _, _, _, _, _, V_ij_tt_1s, log_ll_j_ts, log_ll_ts) \
              = self.filter(ys)
      print "\t\tLog_ll = %s" % str(log_ll_ts[T-1])
      ## Now perform inference
      #(_, S_j_t_x_1Ts, S_jk_tt_1_x_1Ts) = \
      #    self.inference(x_tTs)
      if itr < em_iters:
        means, assignments = kmeans(ys, K)
        S_j_t_x_1Ts = assignment_to_weights(assignments, K)
        Zhat = transition_counts(assignments, K)
        S_jk_tt_1_x_1Ts = tile(Zhat, (T,1,1))
      self.em_update(S_j_t_x_1Ts, S_jk_tt_1_x_1Ts,
          ys, ys, alpha, itr, em_vars)
      W_i_Ts[itr] = S_j_t_x_1Ts
      itr += 1
    return S_jk_tt_1_x_1Ts

  def em_update(self, W_i_T, M_tt_1T, xs, ys, alpha, itr,
      em_vars='all'):
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
      em_vars: Variables to perform learning on
    """
    K, x_dim, y_dim = self.K, self.x_dim, self.y_dim
    (T, _) = shape(xs)
    P_cur_prev = zeros((T,x_dim,x_dim))
    P_cur = zeros((T, x_dim, x_dim))
    means, covars = empirical_wells(xs, W_i_T)
    for t in range(T):
      if t > 0:
        P_cur_prev[t] = outer(xs[t], xs[t-1])
      P_cur[t] = outer(xs[t], xs[t])
    # Update As
    if 'As' in em_vars:
      self.A_update(T, x_dim, W_i_T, M_tt_1T, ys, alpha, covars,
            P_cur_prev, P_cur, itr)
    # Update bs
    if 'bs' in em_vars:
      self.b_update(T, x_dim, W_i_T, M_tt_1T, ys, alpha, means,
          covars, P_cur_prev, P_cur, itr)
    # Update Qs
    if 'Qs' in em_vars:
      self.Q_update(T, x_dim, W_i_T, ys, alpha, itr, covars)
    # Update mus
    if 'mus' in em_vars:
      self.mu_update(T, x_dim, W_i_T, M_tt_1T, ys, alpha, covars,
        P_cur_prev, P_cur, itr)
    # Update Z
    if 'Z' in em_vars:
      self.Z_update(T, x_dim, W_i_T, M_tt_1T,
          alpha, covars, P_cur_prev, P_cur, itr)

  def b_update(self, T, x_dim, W_i_T, M_tt_1T, ys, alpha,
      means, covars, P_cur_prev, P_cur, itr):
    K, x_dim = self.K, self.x_dim
    for i in range(K):
      self.bs[i] = dot(eye(x_dim) - self.As[i], means[i])

  def mu_update(self, T, x_dim, W_i_T, M_tt_1T, xs, alpha, covars,
        P_cur_prev, P_cur, itr):
    K, x_dim = self.K, self.x_dim
    for i in range(K):
      mu_num = zeros(x_dim)
      mu_denom = 0
      mu_num += W_i_T[0,i] * xs[0]
      mu_denom += W_i_T[0,i]
      if mu_denom > 0:
        self.mus[i] = (1/mu_denom) * mu_num

  def Z_update(self, T, x_dim, W_i_T, M_tt_1T, alpha, covars,
        P_cur_prev, P_cur, itr):
    K = self.K
    Z = zeros((K,K))
    for i in range(K):
      for j in range(K):
        Z_denom = 0
        for t in range(1,T):
          Z[i,j] += M_tt_1T[t-1,i,j]
          Z_denom += W_i_T[t,i]
        Z[i,j] /= Z_denom
    for i in range(K):
      s = sum(Z[i,:])
      Z[i,:] /= s
    self.Z = Z

  def A_update(self, T, x_dim, W_i_T, M_tt_1T, ys, alpha, covars,
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

  def Q_update(self, T, x_dim, W_i_T, xs, alpha, itr, covars):
    N_steps = 1
    for step in range(N_steps):
      eta = 2.0/(itr*N_steps + step+2)
      Lambda = alpha * eye(x_dim)
      for i in range(self.K):
        Qdenom = alpha
        Qnum = Lambda
        Anum = zeros((x_dim, x_dim))
        Adenom = zeros((x_dim, x_dim))
        for t in range(1,T):
          x_t = xs[t]
          x_t_pred = dot(self.As[i], xs[t-1]) + self.bs[i]
          diff = x_t - x_t_pred
          Qnum += W_i_T[t,i] * outer(diff, diff)
          Qdenom += W_i_T[t,i]
        Qgrad = -0.5 * Qnum + 0.5 * Qdenom * self.Qs[i]
        Qpred = (1.0/Qdenom) * Qnum
        self.Qs[i] = Qpred

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
  K = dot(pred_V, dot(C.T, linalg.pinv(S)))

  # Update the estimates
  cor_x = pred_x + dot(K, e)
  cor_V = pred_V - dot(K, dot(S, K.T))
  cor_V_joint = dot(eye(dim_state) - dot(K,C), dot(A, V))
  logL = log_multivariate_normal_pdf(y - cor_x,zeros(shape(e)),cor_V)
  return (cor_x, cor_V, cor_V_joint, logL)

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
