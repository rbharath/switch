from cvxopt import matrix, solvers
from numpy import bmat, zeros, reshape, array, dot
from numpy.linalg import pinv
from scipy.linalg import block_diag
# Define constants
a = array([[0.9]])
d = array([[2]])
y = array([1.])
x = array([1.])
b = array([0.5])
v = y - dot(a,x) - b
v = reshape(v, (len(v),1))
B = dot(v,v.T)

def construct_coeff_matrix(x_dim):
  # x = [t vec(Q)]
  g_dim = 1 + 4 * x_dim
  G = zeros((g_dim**2, 1 + x_dim **2))
  # --------------------
  #|t v.T
  #|v  Q
  #|      D-Q    A
  #|      A.T  D^{-1}
  #|                   Q
  # --------------------

  # First Block Column
  G[0,0] = 1.
  # Second Block Column
  left = 1
  top = 1
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = 1 + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = 1.
  # Third Block Column
  left = 1 + x_dim
  top = 1 + x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = 1 + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = -1.
  # Fifth Block Column
  left = 1 + 3 * x_dim
  top = 1 + 3 * x_dim
  for j in range(x_dim): # cols
    for i in range(x_dim): # rows
      vec_pos = 1 + j * x_dim + i #pos in param vector
      mat_pos = left * g_dim + j * g_dim + top + i
      G[mat_pos, vec_pos] = 1.
  return G

def construct_const_matrix(x_dim, y, x, A, D):
  #[[        0.,  y-(a*x+b),  0.,      0., 0.],
  # [ y-(a*x+b),          0., 0.,      0., 0.],
  # [        0,           0.,  D,       A, 0.],
  # [        0,           0.,  A, pinv(D), 0.],
  # [        0,           0., 0.,      0., 0.]]
  # Construct B1
  B1 = zeros((1+x_dim, 1+x_dim))
  x = reshape(x, (x_dim, 1))
  v = y - dot(A,x)
  B1[0,1:] = v.T
  B1[1:,0] = v

  # Construct B2
  B2 = zeros((2 * x_dim, 2 * x_dim))
  B2[0:x_dim, 0:x_dim] = D
  B2[0:x_dim, x_dim:2*x_dim] = A
  B2[x_dim:2*x_dim,0:x_dim] = A.T
  B2[x_dim:2*x_dim,x_dim:2*x_dim] = pinv(D)

  # Construct B3
  B3 = zeros((x_dim, x_dim))

  # Construct Block matrix
  h = block_diag(B1, B2, B3)
  return h

def solve_Q(B, A, D):
  x_dim = len(x)
  c = matrix([1., 0.])
  G = -construct_coeff_matrix(x_dim)
  print "G"
  print G
  Gs = [matrix(G)]

  h = construct_const_matrix(x_dim, y, x, A, D)
  print "h"
  print h
  hs = [matrix(h)]

  sol = solvers.sdp(c, Gs = Gs, hs=hs)
  print sol['x']
  return sol

solve_Q(x, b, y, a, d)
