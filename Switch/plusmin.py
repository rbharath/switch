from SwitchingKalman import *
from numpy import array, reshape, savetxt, loadtxt
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.linalg import svd
import sys

"""The switching system has the following one-dimensional dynamics:
    x_{t+1}^1 = x_t + \epsilon_1
    x_{t+1}^2 = -x_t + \epsilon_2
"""
# Usual
SAMPLE = False
LEARN = True
PLOT = True

## For param changes
#SAMPLE = True
#LEARN = False
#PLOT = True

NUM_ITERS = 5
T = 500
x_dim = 1
y_dim = 1
K = 2
As = reshape(array([[0.5],[0.5]]), (K,x_dim,x_dim))
bs = reshape(array([[0.5],[-0.5]]), (K,x_dim))
Qs = reshape(array([[0.01],[0.01]]), (K,x_dim,x_dim))
Cs = reshape(array([[1],[1]]), (K,y_dim,x_dim))
Rs = reshape(array([[0.01],[0.01]]), (K,y_dim,y_dim))
Z = reshape(array([[0.98, 0.02],[0.02, 0.98]]), (K,K))
pi = reshape(array([0.99,0.01]), (K,))
mus = reshape(array([[1],[-1]]), (K,x_dim))
Sigmas = reshape(array([[0.01],[0.01]]), (K,x_dim,x_dim))
#em_vars = ['As','Qs', 'bs', 'pi', 'Z', 'mus', 'Sigmas', 'Cs', 'Rs']
em_vars = ['As', 'bs', 'Qs', 'Z']
s = SwitchingKalmanFilter(x_dim,y_dim,K=K,As=As,bs=bs,Qs=Qs,Cs=Cs,Rs=Rs,Z=Z, pi=pi,Sigmas=Sigmas, mus=mus)
if SAMPLE:
  xs,Ss,ys = s.sample(T)
  savetxt('../example/xs.txt', xs)
  savetxt('../example/Ss.txt', Ss)
  savetxt('../example/ys.txt', ys)
else:
  xs = reshape(loadtxt('../example/xs.txt'), (T,x_dim))
  Ss = reshape(loadtxt('../example/Ss.txt'), (T))
  ys = reshape(loadtxt('../example/ys.txt'), (T,y_dim))
# Now perform filtering
(M_t_1t, M_tt, W_ij, x_j_tt,x_tt, V_j_tt, V_ij_tt_1,log_ll_j_t,log_ll_t) =\
    s.filter(ys)
true_log_ll = log_ll_t[T-1]
print "True Log-LL = %s" % str(true_log_ll)
(x_j_tT, V_j_tT, U_jk_t, M_tt_1T, M_tT, W_jk_T, x_tT, V_tT) = \
    s.smooth(x_j_tt, V_j_tt, V_ij_tt_1, M_tt, ys)
if LEARN:
  # Compute K-means
  means, assignments = kmeans(ys, K)
  W_i_Ts = assignment_to_weights(assignments,K)
  emp_means, emp_covars = empirical_wells(ys, W_i_Ts)
  for i in range(K):
    A = randn(x_dim, x_dim)
    u, s, v = svd(A)
    As[i] = rand() * dot(u, v.T)
    bs[i] = dot(eye(x_dim) - As[i], means[i])
  l = SwitchingKalmanFilter(x_dim,y_dim,K=K,As=As,bs=bs,Qs=Qs,Cs=Cs,Rs=Rs,Z=Z, pi=pi,Sigmas=Sigmas, mus=mus)
  l.em(ys, em_iters=NUM_ITERS, em_vars=em_vars)
  sim_xs,sim_Ss,sim_ys = l.sample(T,s_init=0, x_init=means[0],
      y_init=means[0])
print "True Log-LL = %s" % str(true_log_ll)
if PLOT:
  plt.close('all')
  plt.figure(1)
  #plt.plot(range(T), xs, label='Hidden State')
  plt.plot(range(T), ys, label="Observations")
  if LEARN:
    plt.plot(range(T), sim_ys, label='Sampled Observations')
  plt.legend()
  plt.show()
